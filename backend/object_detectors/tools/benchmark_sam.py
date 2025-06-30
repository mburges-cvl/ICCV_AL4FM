#!/usr/bin/env python
import os
import json
import argparse
from PIL import Image
import numpy as np
import torch
import rasterio
from tqdm import tqdm
from samgeo.hq_sam import SamGeo
from scipy import ndimage
import pandas as pd
import cv2
import torchvision
import matplotlib.pyplot as plt
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

# --------------------
# Dataset paths (update as needed)
# --------------------
DATASET_PATHS = {
    "DIOR": {
        "train_json": "DIOR/annotations/instances_train2017.json",
        "train_images": "DIOR/train",
        "img_size": 800,
    },
    "SARDet": {
        "train_json": "SARDet-100K/annotations/trainval_split_corrected.json",
        "train_images": "SARDet-100K/train",
        "img_size": 512,
    },
    "Tiny-DOTA": {
        "train_json": "/home/vsz/datasets/Tiny-DOTA/train1024/DOTA2_train1024_tiny_abb.json",
        "train_images": "/home/vsz/datasets/Tiny-DOTA/train1024/images",
        "img_size": 1024,
    },
    "DOTAv2": {
        "train_json": "Tiny-DOTA/train1024/DOTA2_train1024.json",
        "train_images": "Tiny-DOTA/train1024/images",
        "img_size": 1024,
    },
    "VisDrone": {
        "train_json": "/raid/interns_2025/marvin/datasets/VisDrone_squared/annotations/train.json",
        "train_images": "/raid/interns_2025/marvin/datasets/VisDrone_squared/train",
        "img_size": 1024,
    },
    "HRSC2016": {
        "train_json": "/home/vsz/datasets/hrsc2016_squared/annotations/train.json",
        "train_images": "/home/vsz/datasets/hrsc2016_squared/train",
        "img_size": 1024,
    },
    "WaffleHome": {
        "train_json": "/home/vsz/datasets/WaffleHome/train.json",
        "train_images": "/home/vsz/datasets/WaffleHome/images",
        "img_size": 1024,
    },
    "fair1m": {
        "train_json": "/raid/interns_2025/marvin/datasets/fair1m/annotations/train.json",
        "train_images": "/raid/interns_2025/marvin/datasets/fair1m/train",
        "img_size": 1024,
    },
    "coco": {
        "train_json": "/raid/interns_2025/marvin/datasets/minicoco/annotations/instances_minitrain2017.json",
        "train_images": "/raid/interns_2025/marvin/datasets/minicoco/images",
        "img_size": 1024,
    },
    "fineair": {
        "train_json": os.path.join(
            "/raid/interns_2025/marvin/datasets", "fineair", "train.json"
        ),
        "train_images": os.path.join("/raid/interns_2025/marvin/datasets", "fineair"),
        "img_size": 1024,
    },
}


# --------------------
# Helper functions
# --------------------
def normalize_image_mean_std(image, num_std=3):
    """
    Normalize the image using mean ± num_std * std and scale to [0, 255].
    """
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    scale = 255 / (2 * num_std * std)
    normalized_image = (image - (mean - num_std * std)) * scale
    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)
    return normalized_image


def compute_iou(boxA, boxB):
    """
    Compute Intersection-over-Union (IoU) between two boxes.
    Both boxes are given in [x, y, w, h] format.
    """
    # Convert boxes to [x1, y1, x2, y2]
    xA1, yA1, wA, hA = boxA
    xA2, yA2 = xA1 + wA, yA1 + hA
    xB1, yB1, wB, hB = boxB
    xB2, yB2 = xB1 + wB, yB1 + hB

    x_int1 = max(xA1, xB1)
    y_int1 = max(yA1, yB1)
    x_int2 = min(xA2, xB2)
    y_int2 = min(yA2, yB2)

    inter_w = max(0, x_int2 - x_int1)
    inter_h = max(0, y_int2 - y_int1)
    inter_area = inter_w * inter_h

    areaA = wA * hA
    areaB = wB * hB
    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def get_predicted_boxes(img_pil, sam):
    """
    Run SAM on the provided image (as a PIL image) and return a list of
    predicted bounding boxes (each in [x, y, w, h] format).
    """
    image_np = np.array(img_pil)
    with torch.no_grad(), torch.autocast("cuda"):
        sam.generate(image_np, foreground=True, unique=True, hq_token_only=True)
    # Sorted by area (small to large in this example)
    sorted_masks = sorted(sam.masks, key=lambda x: x["area"], reverse=False)
    predicted_boxes = []
    for ann in sorted_masks:
        predicted_boxes.append(ann["bbox"])
    return predicted_boxes


def get_selective_search_boxes(img_pil, mode="fast"):
    """
    Run Selective Search on the provided image (as a PIL image) and return a list of
    region proposals (each in [x, y, w, h] format), sorted by area (small to large).

    Parameters
    ----------
    img_pil : PIL.Image.Image
        The input image.
    mode : str, optional
        "fast" for a quicker but coarser search, or "quality" for a more thorough search.
        Default is "fast".

    Returns
    -------
    List[List[int]]
        List of bounding boxes [x, y, w, h].
    """
    # Convert PIL to OpenCV BGR
    image = np.array(img_pil.convert("RGB"))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create Selective Search segmentation object
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image_bgr)

    # Choose mode
    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    elif mode == "quality":
        ss.switchToSelectiveSearchQuality()
    else:
        raise ValueError("mode must be 'fast' or 'quality'")

    # Run!
    rects = ss.process()  # list of (x, y, w, h) tuples

    max_proposals = 100
    top_rects = rects[:max_proposals]

    # Convert to list format and sort by area
    boxes = [[x, y, w, h] for (x, y, w, h) in top_rects]
    boxes.sort(key=lambda b: b[2] * b[3])  # ascending area

    return boxes


def get_RPN_boxes(img_pil, model):
    """
    Run RPN on the provided image (as a PIL image) and return a list of
    predicted bounding boxes (each in [x, y, w, h] format).
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image_tensor = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)
    boxes = predictions[0]["boxes"].cpu().numpy()

    return boxes


def get_grounded_sam_boxes(img_path, model):
    with torch.no_grad():
        results = model.predict(
            img_path,
        )

    boxes_xyxy = results.xyxy
    boxes_xywh = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        boxes_xywh.append([x1, y1, w, h])
    boxes_xywh = np.array(boxes_xywh)

    return boxes_xywh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SAM dense mask recall using bounding boxes across multiple IoU thresholds and per class."
    )
    parser.add_argument(
        "--dataset_choice",
        type=str,
        default="WaffleHomes",
        help="Choice of dataset (key in DATASET_PATHS)",
    )
    parser.add_argument(
        "--proposal_model",
        type=str,
        default="SAM",
        help="which proposal model to use",
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        help="Type of SAM model (e.g., vit_h)",
    )
    parser.add_argument(
        "--gridsize",
        type=int,
        default=32,
        help="Grid size (points_per_side) for dense mask generation",
    )
    parser.add_argument(
        "--blur_factor",
        type=int,
        default=0,
        help="Blur factor for Gaussian blur (0 to disable)",
    )
    return parser.parse_args()


# --------------------
# Main benchmark function
# --------------------
def main():
    args = parse_args()
    dataset_choice = args.dataset_choice
    sam_model_type = args.sam_model_type
    gridsize = args.gridsize
    proposal_model = args.proposal_model

    if dataset_choice not in DATASET_PATHS:
        raise ValueError(f"Dataset {dataset_choice} not found in DATASET_PATHS.")
    dataset_info = DATASET_PATHS[dataset_choice]
    img_size = dataset_info["img_size"]
    coco_json_path = dataset_info["train_json"]
    image_folder = dataset_info["train_images"]

    # Load COCO JSON (expects keys "images" and "annotations")
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Build a mapping: image_id -> list of GT annotations (each with bbox and category_id)
    gt_boxes_map = {}
    if "annotations" not in coco_data:
        raise ValueError("No annotations found in the provided JSON file.")

    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        category_id = ann.get("category_id", "unknown")
        gt_boxes_map.setdefault(image_id, []).append(
            {"bbox": bbox, "category_id": category_id}
        )

    # If available, build a mapping from category_id to category name.
    cat_id_to_name = {}
    if "categories" in coco_data:
        for cat in coco_data["categories"]:
            cat_id_to_name[cat["id"]] = cat["name"]

    # Define IoU thresholds from 0.5 to 0.95 in steps of 0.05.
    thresholds = np.arange(0.5, 1.0, 0.05)

    # Initialize counters for overall and per class.
    overall_counts = {thr: {"total": 0, "detected": 0} for thr in thresholds}
    per_class_counts = (
        {}
    )  # key: category_id, value: {thr: {"total": count, "detected": count}}

    # Setup SAM.
    if proposal_model == "SAM":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = SamGeo(
            model_type=sam_model_type,
            sam_kwargs={
                "points_per_side": gridsize,
                "points_per_batch": 256,
            },
            hq=True,
            device=device,
        )

    elif proposal_model == "HQSAM":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = SamGeo(
            model_type=sam_model_type,
            sam_kwargs={
                "points_per_side": gridsize,
                "points_per_batch": 256,
            },
            hq=True,
            device=device,
        )
    elif proposal_model == "RPN":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()

    elif proposal_model == "GroundedSAM":
        categories = {}

        for cat in coco_data["categories"]:
            categories[cat["name"]] = cat["name"]

        print(f"Categories: {categories}")

        base_model = GroundedSAM(ontology=CaptionOntology(categories))

    # Process each image.
    total_images = len(coco_data["images"])
    for idx, img_info in enumerate(
        tqdm(coco_data["images"], desc="Benchmarking images")
    ):
        image_id = img_info["id"]
        image_file = img_info["file_name"]
        image_path = os.path.join(image_folder, image_file)

        org_img_size = img_info.get("width", img_size), img_info.get("height", img_size)

        # Load image.
        if image_path.lower().endswith(".tif"):
            with rasterio.open(image_path) as src:
                img = src.read()
                # if img.shape[0] >= 3:
                #     img = np.stack([img[0], img[1], img[2]], axis=-1)
                # else:
                #     img = np.transpose(img, (1, 2, 0))

                img = np.moveaxis(img[[0, 1, 2], :, :], 0, -1)
                img = normalize_image_mean_std(img)
                img_pil = Image.fromarray(img)
        else:
            img_pil = Image.open(image_path).convert("RGB")
            img_np = np.array(img_pil)
            # img_np = normalize_image_mean_std(img_np)
            img_pil = Image.fromarray(img_np)

        img_pil = img_pil.resize((org_img_size[0], org_img_size[1]))
        img_pil_org = img_pil.copy()

        if args.blur_factor > 0:
            img_np = np.array(img_pil)
            img_np = cv2.GaussianBlur(img_np, (args.blur_factor, args.blur_factor), 0)
            img_pil = Image.fromarray(img_np)

        # Get predicted boxes from SAM.
        if proposal_model == "SAM":
            predicted_boxes = get_predicted_boxes(img_pil, sam)

        elif proposal_model == "SelectiveSearch":
            predicted_boxes = get_selective_search_boxes(img_pil, mode="fast")

        elif proposal_model == "RPN":
            predicted_boxes = get_RPN_boxes(img_pil, model)

        elif proposal_model == "GroundedSAM":
            predicted_boxes = get_grounded_sam_boxes(image_path, base_model)

        # For each ground truth annotation in this image, compute its maximum IoU with any prediction.
        gt_annotations = gt_boxes_map.get(image_id, [])
        for ann in gt_annotations:
            gt_box = ann["bbox"]
            category_id = ann["category_id"]

            # Compute maximum IoU over all predicted boxes.
            max_iou = 0.0
            for pred_box in predicted_boxes:
                iou = compute_iou(gt_box, pred_box)
                if iou > max_iou:
                    max_iou = iou

            # Update overall and per-class counts for each threshold.
            for thr in thresholds:
                overall_counts[thr]["total"] += 1
                if max_iou >= thr:
                    overall_counts[thr]["detected"] += 1

                if category_id not in per_class_counts:
                    per_class_counts[category_id] = {
                        t: {"total": 0, "detected": 0} for t in thresholds
                    }
                per_class_counts[category_id][thr]["total"] += 1
                if max_iou >= thr:
                    per_class_counts[category_id][thr]["detected"] += 1

        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # ax[0].imshow(img_pil_org)
        # ax[0].set_title("Image")
        # ax[1].imshow(img_pil)
        # ax[1].set_title("Predicted Boxes")
        # for box in predicted_boxes:
        #     x, y, w, h = box
        #     rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
        #     ax[1].add_patch(rect)
        # for ann in gt_annotations:
        #     gt_box = ann["bbox"]
        #     x, y, w, h = gt_box
        #     rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor="g", facecolor="none")
        #     ax[1].add_patch(rect)

        # ax[1].set_title("Predicted Boxes with GT")
        # plt.tight_layout()
        # plt.savefig(f"sam_results/{image_file}_predicted_boxes.png")
        # plt.close(fig)

        # exit()

        # print every 10% based on total_images
        if idx % (total_images // 10) == 0:
            print(
                f"Processed {idx}/{total_images} images ({(idx / total_images) * 100:.2f}%)"
            )
            print("Intermediate results:")
            for thr in thresholds:
                total = overall_counts[thr]["total"]
                detected = overall_counts[thr]["detected"]
                recall = detected / total if total > 0 else 0.0
                print(
                    f"IoU ≥ {thr:.2f} - Overall Recall: {recall:.4f} ({detected}/{total})"
                )
            print("-" * 50)

        # break

    # Compute recall for overall and per class.
    overall_results = []
    for thr in thresholds:
        total = overall_counts[thr]["total"]
        detected = overall_counts[thr]["detected"]
        recall = detected / total if total > 0 else 0.0
        overall_results.append(
            {
                "dataset": dataset_choice,
                "sam_model": sam_model_type,
                "gridsize": gridsize,
                "iou_threshold": thr,
                "class": "overall",
                "total_gt": total,
                "detected_gt": detected,
                "recall": recall,
            }
        )
        print(f"IoU ≥ {thr:.2f} - Overall Recall: {recall:.4f} ({detected}/{total})")

    per_class_results = []
    for cat_id, counts in per_class_counts.items():
        cat_name = cat_id_to_name.get(cat_id, str(cat_id))
        for thr in thresholds:
            total = counts[thr]["total"]
            detected = counts[thr]["detected"]
            recall = detected / total if total > 0 else 0.0
            per_class_results.append(
                {
                    "dataset": dataset_choice,
                    "sam_model": sam_model_type,
                    "gridsize": gridsize,
                    "iou_threshold": thr,
                    "class": cat_name,
                    "total_gt": total,
                    "detected_gt": detected,
                    "recall": recall,
                }
            )
            print(
                f"IoU ≥ {thr:.2f} - Class '{cat_name}' Recall: {recall:.4f} ({detected}/{total})"
            )

    # Combine overall and per class results.
    all_results = overall_results + per_class_results

    # Save results as .csv.
    results_df = pd.DataFrame(all_results)
    output_csv = f"sam_results/results_{dataset_choice}_{sam_model_type}_{gridsize}_{proposal_model}_{args.blur_factor}_iou.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()
