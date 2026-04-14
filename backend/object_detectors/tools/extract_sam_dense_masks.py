import argparse
import os
import json
from PIL import Image
import rasterio
import numpy as np
from tqdm import trange
import torch
from scipy import ndimage
from samgeo.hq_sam import (
    SamGeo,
    show_image,
    download_file,
    overlay_images,
    tms_to_geotiff,
)


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
            "Tiny_DOTA",
            "visdrone",
            "HRSC2016",
            "WaffleHome",
            "DOTAv2",
            "fair1m",
            "coco",
            "fineair",
            "CHAI",
        ],
        help="Dataset to use (default: 'Tiny-DOTA')",
    )
    parser.add_argument(
        "--pixels_between_points",
        type=int,
        default=16,
        help="Spacing between points for mask generation (default: 16)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="cvl",
        choices=["dgx", "hpe", "cvl"],
        help=(
            "Server type: 'dgx' or 'hpe'. For dgx, outdir is '/raid/interns_2025/marvin/' and "
            "datasets are stored in '/raid/interns_2025/marvin/datasets/'; for hpe, outdir is "
            "'/data/interns_2025/marvin/' and datasets are stored in '/home/vsz/datasets/'."
        ),
    )
    parser.add_argument(
        "--sam_test",
        action="store_true",
        default=False,
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
        "Tiny_DOTA": {
            "train_json": os.path.join(
                dataset_base, "Tiny-DOTA", "train1024", "DOTA2_train1024_tiny_abb.json"
            ),
            "train_images": os.path.join(
                dataset_base, "Tiny-DOTA", "train1024", "images"
            ),
            "img_size": 1024,
        },
        "visdrone": {
            "train_json": os.path.join(
                dataset_base, "visdrone_squared", "annotations", "train.json"
            ),
            "train_images": os.path.join(dataset_base, "visdrone_squared", "train"),
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
        "coco": {
            "train_json": os.path.join(
                dataset_base, "minicoco", "annotations", "instances_minitrain2017.json"
            ),
            "train_images": os.path.join(dataset_base, "minicoco", "images"),
            "img_size": 640,
        },
        "fineair": {
            "train_json": os.path.join(dataset_base, "fineair", "train.json"),
            "train_images": os.path.join(dataset_base, "fineair"),
            "img_size": 1024,
        },
        "CHAI": {
            "train_json": os.path.join(dataset_base, "IOD_Datasets/CHAI/annotations", "instances_train2017.json"),
            "train_images": os.path.join(dataset_base, "IOD_Datasets/CHAI", "images"),
            "img_size": 1024,
        },
    }


def keep_largest_component(mask_2d):
    """
    Keeps only the largest connected component in a 2D binary mask.
    """
    labeled_array, num_components = ndimage.label(mask_2d)
    if num_components <= 1:
        return mask_2d
    component_sizes = np.bincount(labeled_array.ravel())
    largest_label = component_sizes[1:].argmax() + 1
    filtered_mask = (labeled_array == largest_label).astype(mask_2d.dtype)
    return filtered_mask


def filter_masks_and_boxes(masks, boxes):
    """
    For each object in `masks`, keep only the largest connected component
    and update `boxes` accordingly.
    """
    H, W, N = masks.shape
    for i in range(N):
        mask_2d = masks[..., i]
        filtered_2d = keep_largest_component(mask_2d)
        masks[..., i] = filtered_2d
        coords = np.argwhere(filtered_2d > 0)
        if len(coords) == 0:
            boxes[i] = [0, 0, 0, 0]
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            boxes[i] = [x_min, y_min, w, h]
    return masks, boxes


def normalize_image_mean_std(image, num_std=3):
    """
    Normalize the image using mean ± num_std * std and scale to [0, 255].
    Handles cases where std is zero to avoid division by zero errors.
    """
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)

    # Handle zero or near-zero standard deviations to avoid division by zero
    # Add a small epsilon to prevent division by zero
    epsilon = 1e-6
    std = np.maximum(std, epsilon)

    scale = 255 / (2 * num_std * std)
    normalized_image = (image - (mean - num_std * std)) * scale
    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)
    return normalized_image


import time


def get_and_save_masks(image_id, img_pil, mask_dir, sam):
    total_start = time.time()

    # --- Step 1: Generate masks using SAM with fp16 ---
    t0 = time.time()
    with torch.no_grad() and torch.autocast("cuda"):
        sam.generate(np.array(img_pil), foreground=True, unique=True)
    t1 = time.time()
    print(f"[{image_id}] SAM.generate took: {t1 - t0:.4f} sec")

    # --- Step 2: Retrieve and sort masks ---
    t0 = time.time()
    point_grids = sam.mask_generator.point_grids  # Not used further here, but timed
    masks = sam.masks
    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=False)
    t1 = time.time()
    print(f"[{image_id}] Retrieving and sorting masks took: {t1 - t0:.4f} sec")

    if len(sorted_masks) == 0:
        print(f"[{image_id}] No masks found.")
        return np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint16), np.zeros(
            (0, 4)
        )

    # --- Step 3: Build objects_stacked array and collect boxes ---
    t0 = time.time()
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
    t1 = time.time()
    print(f"[{image_id}] Building objects_stacked array took: {t1 - t0:.4f} sec")

    # --- Step 4: Filter masks and update boxes ---
    t0 = time.time()
    objects_stacked, boxes = filter_masks_and_boxes(objects_stacked, boxes)
    t1 = time.time()
    print(f"[{image_id}] Filtering masks and boxes took: {t1 - t0:.4f} sec")

    # --- Step 5: Post-processing segmentation and boxes ---
    t0 = time.time()
    seg = objects_stacked.transpose(2, 0, 1)
    idx_array = np.arange(seg.shape[0]).reshape(-1, 1, 1) + 1
    mask_bool = seg.astype(bool)
    objects = np.sum(mask_bool * idx_array, axis=0)
    dtype = np.uint16
    objects = objects.astype(dtype)
    boxes = np.array(boxes)
    t1 = time.time()
    print(f"[{image_id}] Post-processing took: {t1 - t0:.4f} sec")

    total_end = time.time()
    print(
        f"[{image_id}] Total time for get_and_save_masks: {total_end - total_start:.4f} sec"
    )
    return objects, boxes


def main():
    args = parse_args()

    SAM_MODEL_TYPE = args.sam_model_type
    DATASET_CHOICE = args.dataset
    pixels_between_points = args.pixels_between_points
    server = args.server.lower()

    # Set dataset base path and output directory based on the server type
    if server == "dgx":
        dataset_base = "/raid/interns_2025/marvin/datasets"
        outdir = "/raid/interns_2025/marvin/"
    elif server == "hpe":
        dataset_base = "/home/vsz/datasets"
        outdir = "/data/interns_2025/marvin/"
    elif server == "cvl":
        dataset_base = "/data/mburges/datasets"
        outdir = "/caa/Homes01/mburges/ICCV_AL4FM/"
    else:
        raise ValueError("Server must be either 'dgx' or 'hpe'.")

    # Build the dataset paths based on the dataset base
    DATASET_PATHS = get_dataset_paths(dataset_base)

    img_size = DATASET_PATHS[DATASET_CHOICE]["img_size"]
    points_per_side = img_size // pixels_between_points
    device = "cuda" if torch.cuda.is_available() else "cpu"

    coco_json_path = DATASET_PATHS[DATASET_CHOICE]["train_json"]
    image_folder = DATASET_PATHS[DATASET_CHOICE]["train_images"]

    if args.sam_test:
        coco_json_path = coco_json_path.replace("train", "test")
        image_folder = image_folder.replace("train", "test")

    dataset_dir = os.path.dirname(image_folder)
    mask_dir = os.path.join(dataset_dir, "train_masks")
    os.makedirs(mask_dir, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    sam = SamGeo(
        model_type=SAM_MODEL_TYPE,
        sam_kwargs={
            "points_per_side": points_per_side,
            "points_per_batch": 128,
        },
        hq=True,
        device=device,
    )

    all_objects = []
    all_boxes = []
    all_img_idx = []

    assert os.path.exists(outdir), f"Output directory {outdir} does not exist."

    for i in trange(len(coco_data["images"])):
        print()
        img_info = coco_data["images"][i]
        image_id = img_info["id"]
        image_path = os.path.join(image_folder, img_info["file_name"])

        if image_path.endswith(".tif"):
            with rasterio.open(image_path) as src:
                img = src.read()
                # img = np.moveaxis(img[[4, 2, 1], :, :], 0, -1)
                # img = np.moveaxis(img[[0,1,2], :, :], 0, -1)
                img = np.moveaxis(img[[0, 1, 2], :, :], 0, -1)
                img = normalize_image_mean_std(img)
                img_pil = Image.fromarray(img)
        else:
            img_pil = Image.open(image_path).convert("RGB")
            # img = normalize_image_mean_std(np.array(img_pil))
            # img_pil = Image.fromarray(img)

        img_pil = img_pil.resize((img_size, img_size))
        objects, boxes = get_and_save_masks(image_id, img_pil, mask_dir, sam)
        img_ids_f_boxes = np.ones((boxes.shape[0], 1)) * image_id

        all_objects.append(objects)
        all_boxes.append(boxes)
        all_img_idx.append(img_ids_f_boxes)

        # if i > 10:
        #     exit()

    all_objects = np.stack(all_objects)
    all_boxes = np.vstack(all_boxes)
    all_img_idx = np.vstack(all_img_idx)

    print(all_objects.shape, all_boxes.shape, all_img_idx.shape)

    if args.sam_test:
        addon = "_test"
    else:
        addon = ""

    np.save(
        os.path.join(
            outdir,
            f"objects_{DATASET_CHOICE.lower()}_{pixels_between_points}{addon}.npy",
        ),
        all_objects,
    )
    np.save(
        os.path.join(
            outdir, f"boxes_{DATASET_CHOICE.lower()}_{pixels_between_points}{addon}.npy"
        ),
        all_boxes,
    )
    np.save(
        os.path.join(
            outdir,
            f"img_idx_{DATASET_CHOICE.lower()}_{pixels_between_points}{addon}.npy",
        ),
        all_img_idx,
    )


if __name__ == "__main__":
    main()
