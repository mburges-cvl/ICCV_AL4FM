import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from samgeo.hq_sam import SamGeo as HQSamGeo
import json
from pycocotools.coco import COCO
import numpy as np
from tqdm import trange
from scipy import ndimage
from PIL import Image
import rasterio
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SAM extraction framework for ORNL")
    parser.add_argument("--img_size", type=int, default=1024, help="Image size")
    parser.add_argument(
        "--sam_model_type", type=str, default="vit_h", help="SAM model type"
    )
    parser.add_argument(
        "--dataset", type=str, default="hrsc2016_squared", help="Dataset name"
    )
    parser.add_argument(
        "--img_ids", type=str, default=None, help="Image IDs to process"
    )

    return parser.parse_args()


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


def filter_masks_and_boxes(masks, boxes=None):
    """
    For each object in `masks`, keep only the largest connected component
    and update `boxes` accordingly.
    """
    H, W, N = masks.shape
    for i in range(N):
        mask_2d = masks[..., i]
        filtered_2d = keep_largest_component(mask_2d)
        masks[..., i] = filtered_2d

        if boxes is None:
            continue

        coords = np.argwhere(filtered_2d > 0)
        if len(coords) == 0:
            boxes[i] = [0, 0, 0, 0]
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            boxes[i] = [x_min, y_min, w, h]

    if boxes is None:
        return masks
    else:
        return masks, boxes


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


def get_and_save_masks_dense(sam, img_pil, image_id):

    with torch.no_grad() and torch.autocast("cuda"):
        sam.generate(np.array(img_pil), foreground=True, unique=True)

    point_grids = sam.mask_generator.point_grids  # Not used further here, but timed
    masks = sam.masks
    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=False)

    if len(sorted_masks) == 0:
        print(f"[{image_id}] No masks found.")
        return np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint16), np.zeros(
            (0, 4)
        )

    objects_stacked = np.zeros(
        (
            sorted_masks[0]["segmentation"].shape[0],
            sorted_masks[0]["segmentation"].shape[1],
            len(sorted_masks),
        )
    )
    boxes = []
    for index, ann in enumerate(sorted_masks):
        m = ann["segmentation"]
        boxes.append(ann["bbox"])
        objects_stacked[:, :, index] = m

    objects_stacked, boxes = filter_masks_and_boxes(objects_stacked, boxes)

    seg = objects_stacked.transpose(2, 0, 1)
    idx_array = np.arange(seg.shape[0]).reshape(-1, 1, 1) + 1
    mask_bool = seg.astype(bool)
    objects = np.sum(mask_bool * idx_array, axis=0)
    dtype = np.uint16
    objects = objects.astype(dtype)
    boxes = np.array(boxes)

    return objects, boxes


def get_and_save_masks(sam, img_pil, bboxes):

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
    # overlaps = np.sum(mask_bool, axis=0)

    dtype = np.uint16
    objects = objects.astype(dtype)
    # overlaps = overlaps.astype(dtype)

    return objects


def extract_dense_masks(
    sam_model_type,
    coco_json,
    img_dir,
    output_dir,
    dataset_name,
    pixels_between_points=16,
    img_size=1024,
    parent=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    points_per_side = img_size // pixels_between_points

    sam = HQSamGeo(
        model_type=sam_model_type,
        sam_kwargs={
            "points_per_side": points_per_side,
            "points_per_batch": 128,
        },
        hq=True,
        device=device,
    )

    coco = COCO(coco_json)

    available_img_ids = coco.getImgIds()

    parent.extraction_progress["total"] = len(available_img_ids)
    parent.extraction_progress["current"] = 0
    parent.extraction_progress["message"] = (
        f"Processing 0/{len(available_img_ids)} images"
    )

    print(f"Total number of images: {len(available_img_ids)}")
    print(coco_json)

    all_objects = []
    all_boxes = []
    all_img_idx = []

    for img_id in tqdm(available_img_ids):
        img_info = coco.loadImgs(img_id)[0]
        image_id = img_info["id"]

        assert image_id == img_id

        image_path = os.path.join(img_dir, img_info["file_name"])

        if image_path.endswith(".tif"):
            with rasterio.open(image_path) as src:
                img = src.read()

                if img.shape[0] == 4:
                    img = np.stack([img[3], img[2], img[1]], axis=-1)
                elif img.shape[0] == 3:
                    img = np.stack([img[2], img[1], img[0]], axis=-1)
                elif img.shape[0] == 8:
                    img = np.stack([img[4], img[2], img[1]], axis=-1)
                img = normalize_image_mean_std(img)
                img_pil = Image.fromarray(img)
        else:
            img_pil = Image.open(image_path)
            img = normalize_image_mean_std(np.array(img_pil))
            img_pil = Image.fromarray(img)

        objects, boxes = get_and_save_masks_dense(sam, img_pil, image_id)

        img_ids_f_boxes = np.ones((boxes.shape[0], 1)) * image_id

        parent.extraction_progress["current"] += 1
        parent.extraction_progress["message"] = (
            f"Processing {parent.extraction_progress['current']}/{parent.extraction_progress['total']} images"
        )

        all_objects.append(objects)
        all_boxes.append(boxes)
        all_img_idx.append(img_ids_f_boxes)

    all_objects = np.stack(all_objects)
    all_boxes = np.vstack(all_boxes)
    all_img_idx = np.vstack(all_img_idx)

    np.save(
        os.path.join(
            output_dir, f"objects_{dataset_name.lower()}_{pixels_between_points}.npy"
        ),
        all_objects,
    )
    np.save(
        os.path.join(
            output_dir, f"boxes_{dataset_name.lower()}_{pixels_between_points}.npy"
        ),
        all_boxes,
    )
    np.save(
        os.path.join(
            output_dir, f"img_idx_{dataset_name.lower()}_{pixels_between_points}.npy"
        ),
        all_img_idx,
    )


def extract_gt_masks(sam_model_type, coco_json, img_dir, output_dir, dataset_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = HQSamGeo(
        model_type=sam_model_type,
        automatic=False,
        hq=True,
        device=device,
    )

    coco = COCO(coco_json)

    with open(coco_json, "r") as f:
        coco_data = json.load(f)

    available_img_ids = coco.getImgIds()

    print(f"Total number of images: {len(available_img_ids)}")

    all_objects = []

    for img_id in tqdm(available_img_ids):
        img_info = coco.loadImgs(img_id)[0]
        image_id = img_info["id"]

        assert image_id == img_id

        image_path = os.path.join(img_dir, img_info["file_name"])

        bboxes = []
        for ann in coco_data["annotations"]:
            if ann["image_id"] == image_id:
                # COCO bbox format = [x, y, w, h]
                bboxes.append(ann["bbox"])

        if len(bboxes) == 0:
            # No GT boxes => skip or handle specially
            # continue
            objects = np.zeros((1024, 1024), dtype=np.uint16)
            all_objects.append(objects)
            continue

        if image_path.endswith(".tif"):
            with rasterio.open(image_path) as src:
                img = src.read()

                if img.shape[0] == 4:
                    img = np.stack([img[3], img[2], img[1]], axis=-1)
                elif img.shape[0] == 3:
                    img = np.stack([img[2], img[1], img[0]], axis=-1)
                elif img.shape[0] == 8:
                    img = np.stack([img[4], img[2], img[1]], axis=-1)
                img = normalize_image_mean_std(img)
                img_pil = Image.fromarray(img)
        else:
            img_pil = Image.open(image_path)
            img = normalize_image_mean_std(np.array(img_pil))
            img_pil = Image.fromarray(img)

        objects = get_and_save_masks(sam, img_pil, bboxes)

        all_objects.append(objects)
        # all_overlaps.append(overlaps)

    all_objects = np.stack(all_objects)

    print(all_objects.shape)
    np.save(os.path.join(output_dir, "objects_al_gt.npy"), all_objects)


if __name__ == "__main__":
    args = parse_args()

    IMG_SIZE = args.img_size
    SAM_MODEL_TYPE = args.sam_model_type
    DATASET = args.dataset
    ORACLE_JSON = f"/home/vsz/datasets/{DATASET}/annotations.json"
    ORACLE_IMAGES = f"/home/vsz/datasets/{DATASET}/images"
    MASKS_DIR = f"/home/vsz/datasets/{DATASET}/masks_gt_al.npy"
    IMG_IDS = args.img_ids
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = HQSamGeo(
        model_type=SAM_MODEL_TYPE,
        automatic=False,
        hq=True,
        device=device,
    )

    coco = COCO(ORACLE_JSON)

    with open(ORACLE_JSON, "r") as f:
        coco_data = json.load(f)

    available_img_ids = coco.getImgIds()

    if IMG_IDS is None:
        new_img_ids = available_img_ids[:200]

    all_objects = []
    # all_overlaps = []

    for img_id in tqdm(new_img_ids):
        img_info = coco.loadImgs(img_id)[0]
        image_id = img_info["id"]

        assert image_id == img_id

        image_path = os.path.join(ORACLE_IMAGES, img_info["file_name"])

        bboxes = []
        for ann in coco_data["annotations"]:
            if ann["image_id"] == image_id:
                # COCO bbox format = [x, y, w, h]
                bboxes.append(ann["bbox"])

        if len(bboxes) == 0:
            # No GT boxes => skip or handle specially
            continue

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

        objects = get_and_save_masks(img_pil, bboxes)

        all_objects.append(objects)
        # all_overlaps.append(overlaps)

    all_objects = np.stack(all_objects)
    # all_overlaps = np.stack(all_overlaps)

    print(all_objects.shape)
    # np.save(MASKS_DIR, all_objects)
