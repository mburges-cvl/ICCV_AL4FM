from samgeo.hq_sam import (
    SamGeo,
    show_image,
    download_file,
    overlay_images,
    tms_to_geotiff,
)
import os
import json
from PIL import Image
import rasterio
import numpy as np
from tqdm import trange
import torch
from scipy import ndimage
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM Geo mask generation with configurable parameters."
    )
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        help="Type of SAM model to use (default: 'vit_h')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Tiny-DOTA",
        choices=[
            "DIOR",
            "Tiny-DOTA",
            "VisDrone",
            "HRSC2016",
            "WaffleHome",
            "fair1m",
            "DOTAv2",
            "fineair",
        ],
        help="Dataset to use (default: 'Tiny-DOTA')",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="dgx",
        choices=["dgx", "hpe"],
        help=(
            "Server type: 'dgx' or 'hpe'. For dgx, outdir is '/raid/interns_2025/marvin/' and "
            "datasets are stored in '/raid/interns_2025/marvin/datasets/'; for hpe, outdir is "
            "'/data/interns_2025/marvin/' and datasets are stored in '/home/vsz/datasets/'."
        ),
    )
    return parser.parse_args()


def get_dataset_paths(dataset_base):
    return {
        "DIOR": {
            "train_json": os.path.join(
                dataset_base, "DIOR", "annotations", "train.json"
            ),
            "train_images": os.path.join(dataset_base, "DIOR", "images", "train"),
            "img_size": 800,
        },
        "Tiny-DOTA": {
            "train_json": os.path.join(
                dataset_base, "Tiny-DOTA", "train1024", "DOTA2_train1024_tiny_abb.json"
            ),
            "train_images": os.path.join(
                dataset_base, "Tiny-DOTA", "train1024", "images"
            ),
            "img_size": 1024,
        },
        "VisDrone": {
            "train_json": os.path.join(
                dataset_base, "VisDrone_squared", "annotations", "train.json"
            ),
            "train_images": os.path.join(dataset_base, "VisDrone_squared", "train"),
            "img_size": 1024,
        },
        "HRSC2016": {
            "train_json": os.path.join(
                dataset_base, "hrsc2016_squared", "annotations", "train.json"
            ),
            "train_images": os.path.join(dataset_base, "hrsc2016_squared", "train"),
            "img_size": 1024,
        },
        "WaffleHome": {
            "train_json": os.path.join(dataset_base, "WaffleHome", "train.json"),
            "train_images": os.path.join(dataset_base, "WaffleHome", "images"),
            "img_size": 1024,
        },
        "DOTAv2": {
            "train_json": os.path.join(
                dataset_base, "Tiny-DOTA", "train1024", "DOTA2_train1024.json"
            ),
            "train_images": os.path.join(
                dataset_base, "Tiny-DOTA", "train1024", "images"
            ),
            "img_size": 1024,
        },
        "fair1m": {
            "train_json": os.path.join(
                dataset_base, "fair1m", "annotations", "train.json"
            ),
            "train_images": os.path.join(dataset_base, "fair1m", "train"),
            "img_size": 1024,
        },
        "fineair": {
            "train_json": os.path.join(dataset_base, "fineair", "train.json"),
            "train_images": os.path.join(dataset_base, "fineair"),
            "img_size": 1024,
        },
    }


def keep_largest_component(mask_2d):
    """
    Keeps only the largest connected component in a 2D binary mask.
    """
    labeled_array, num_components = ndimage.label(mask_2d)
    if num_components <= 1:
        # Either 0 or 1 component: nothing to filter
        return mask_2d

    # Count pixels in each component ID
    # index 0 is background, so skip it when looking for max
    component_sizes = np.bincount(labeled_array.ravel())
    largest_label = component_sizes[1:].argmax() + 1

    # Rebuild the filtered 2D mask with only that largest component
    filtered_mask = (labeled_array == largest_label).astype(mask_2d.dtype)
    return filtered_mask


def filter_masks_and_boxes(masks):
    """
    For each object in `masks`, keep only the largest connected component
    and update `boxes` accordingly.

    Args:
        masks (np.ndarray): shape (H, W, N), binary masks for each object
        boxes (np.ndarray): shape (N, 4), bounding boxes in (x1, y1, x2, y2) format

    Returns:
        (masks, boxes): the filtered masks and updated bounding boxes.
    """
    H, W, N = masks.shape
    for i in range(N):
        # 1) Keep only the largest component in masks[..., i]
        mask_2d = masks[..., i]
        filtered_2d = keep_largest_component(mask_2d)
        masks[..., i] = filtered_2d

    masks = masks.astype(int)

    return masks


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


def get_and_save_masks(image_id, img_pil, mask_dir, bboxes, sam):

    boxes_xyxy = []

    for b_idx, bbox in enumerate(bboxes):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]

        scaled_box = [x1, y1, x2, y2]
        boxes_xyxy.append(scaled_box)

    sam.set_image(np.array(img_pil))
    masks_list = []
    for scaled_box in boxes_xyxy:
        # We call sam.predict() once per box
        masks, scores, low_res_logits = sam.predict(
            boxes=[scaled_box], return_results=True
        )
        masks = masks[0]

        masks_list.append(masks)

    sorted_masks = sorted(masks_list, key=(lambda x: (x == 1).sum()), reverse=True)

    objects_stacked = np.stack(sorted_masks, axis=-1)

    objects_stacked = filter_masks_and_boxes(objects_stacked)

    seg = objects_stacked.transpose(2, 0, 1)
    idx_array = np.arange(seg.shape[0]).reshape(-1, 1, 1) + 1
    # Boolean mask indicating where seg is True
    mask_bool = seg.astype(bool)

    # Update objects
    objects = np.sum(mask_bool * idx_array, axis=0)

    # Update overlaps
    overlaps = np.sum(mask_bool, axis=0)

    dtype = np.uint16
    objects = objects.astype(dtype)
    overlaps = overlaps.astype(dtype)

    return objects, overlaps


def main():
    args = parse_args()

    SAM_MODEL_TYPE = args.sam_model_type
    DATASET_CHOICE = args.dataset
    server = args.server.lower()

    # Set dataset base path and output directory based on the server type
    if server == "dgx":
        dataset_base = "/raid/interns_2025/marvin/datasets"
        outdir = "/raid/interns_2025/marvin/"
    elif server == "hpe":
        dataset_base = "/home/vsz/datasets"
        outdir = "/data/interns_2025/marvin/"
    else:
        raise ValueError("Server must be either 'dgx' or 'hpe'.")

    # Build the dataset paths based on the dataset base
    DATASET_PATHS = get_dataset_paths(dataset_base)

    img_size = DATASET_PATHS[DATASET_CHOICE]["img_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    coco_json_path = DATASET_PATHS[DATASET_CHOICE]["train_json"]
    image_folder = DATASET_PATHS[DATASET_CHOICE]["train_images"]

    dataset_dir = os.path.dirname(image_folder)
    mask_dir = os.path.join(dataset_dir, "masks_gt")
    os.makedirs(mask_dir, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    sam = SamGeo(
        model_type=SAM_MODEL_TYPE,
        automatic=False,
        hq=True,
        device=device,
    )

    all_objects = []
    all_overlaps = []

    assert os.path.exists(outdir), f"Output directory {outdir} does not exist."

    print(f"Extracting masks for {DATASET_CHOICE} dataset...", len(coco_data["images"]))

    for i in trange(len(coco_data["images"])):
        img_info = coco_data["images"][i]
        image_id = img_info["id"]
        image_path = os.path.join(image_folder, img_info["file_name"])

        bboxes = []
        for ann in coco_data["annotations"]:
            if ann["image_id"] == image_id:
                # COCO bbox format = [x, y, w, h]
                bboxes.append(ann["bbox"])

        if len(bboxes) == 0:
            # No GT boxes => skip or handle specially
            objects = np.zeros((img_size, img_size), dtype=np.uint16)
            overlaps = np.zeros((img_size, img_size), dtype=np.uint16)

            all_objects.append(objects)
            all_overlaps.append(overlaps)
            continue

        # Reload for plotting (optional if you don't need the original image)
        if image_path.endswith(".tif"):
            with rasterio.open(image_path) as src:
                img = src.read()
                img = np.stack([img[0], img[1], img[2]], axis=-1)
                img = normalize_image_mean_std(img)
                img_pil = Image.fromarray(img)
        else:
            img_pil = Image.open(image_path)
            img = normalize_image_mean_std(np.array(img_pil))
            img_pil = Image.fromarray(img)

        img_pil = img_pil.resize((img_size, img_size))

        objects, overlaps = get_and_save_masks(image_id, img_pil, mask_dir, bboxes, sam)

        all_objects.append(objects)
        all_overlaps.append(overlaps)

    all_objects = np.stack(all_objects)
    all_overlaps = np.stack(all_overlaps)

    print(all_objects.shape, all_overlaps.shape)

    np.save(
        os.path.join(outdir, f"objects_{DATASET_CHOICE.lower()}_gt.npy"), all_objects
    )


if __name__ == "__main__":
    main()
