from matplotlib import pyplot as plt
from PIL import Image
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math
from .matcher import HungarianMatcher


# Example usage in get_uncertain_images_box:
def get_uncertain_images_box(
    n_images,
    coco,
    selected_image_ids,
    predictions,
    sam_boxes,
    box_to_idx,
    device="cuda",
    alpha=1.0,
    beta=0.5,
    uncertainty_method="original",
    score_threshold=0.5,
):
    """
    For each available image, compute the uncertainty score and then select the top n_images
    with the highest uncertainty.
    """
    weight_dict = {"cost_conf": 1, "cost_bbox": 1, "cost_giou": 1}
    matcher = HungarianMatcher(weight_dict=weight_dict)

    all_img_ids = coco.getImgIds()
    available_img_ids = list(set(all_img_ids) - set(selected_image_ids))
    uncertainties = []
    indices_list = []

    for idx, image_id in tqdm(
        enumerate(available_img_ids), total=len(available_img_ids)
    ):
        sam_boxes_idx = (box_to_idx == image_id).squeeze()
        sam_boxes_image = sam_boxes[sam_boxes_idx]

        img_predictions = predictions[str(image_id)]
        boxes = np.array(img_predictions["boxes"])
        # convert boxes from xyxy to xywh
        boxes[:, 2:] -= boxes[:, :2]
        scores = np.array(img_predictions["scores"])

        boxes_torch = torch.from_numpy(boxes).float()
        scores_torch = torch.from_numpy(scores).float()
        sam_boxes_torch = torch.from_numpy(sam_boxes_image).float()

        # Filter out low-score predictions.
        boxes_torch = boxes_torch[scores_torch > score_threshold]
        scores_torch = scores_torch[scores_torch > score_threshold]

        predictions_ = {"pred_boxes": boxes_torch, "pred_scores": scores_torch}
        targets = {"boxes": sam_boxes_torch}

        cost_matrix_np, giou_matrix_np, indices = matcher(predictions_, targets)
        # Use the new intelligent uncertainty computation.
        uncertainty, U, indices = HungarianMatcher.compute_uncertainty_intelligent(
            cost_matrix_np,
            scores_torch,
            alpha=alpha,
            beta=beta,
            uncertainty_method=uncertainty_method,
            giou_matrix_np=giou_matrix_np,
        )
        uncertainties.append(uncertainty)
        indices_list.append(indices)

    uncertainties = np.array(uncertainties)
    # You may want to select images with the highest uncertainty.
    sorted_indices = np.argsort(uncertainties)
    sorted_img_ids = np.array(available_img_ids)[sorted_indices]
    selected_img_ids = sorted_img_ids[-n_images:]
    selected_indices = [indices_list[i] for i in sorted_indices[-n_images:]]
    # sorted_indices = np.argsort(uncertainties)
    # sorted_img_ids = np.array(available_img_ids)[sorted_indices]
    # selected_img_ids = sorted_img_ids[:n_images]

    return selected_img_ids.tolist(), selected_indices
