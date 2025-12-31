"""
FAST Active Learning Sampling Method.

FAST (Feature-Aware Sample selection through Tracking) combines uncertainty
and diversity sampling for efficient active learning in object detection.

This implementation supports:
- Uncertainty-based sampling using model predictions
- Diversity-based sampling using feature space distances
- Combined sampling with configurable alpha/beta weights
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch
from pycocotools.coco import COCO


def compute_uncertainty_scores(
    predictions: dict,
    img_ids: List[int],
    method: str = "exponential",
) -> dict:
    """
    Compute uncertainty scores for each image based on model predictions.

    Args:
        predictions: Dict mapping image_id to list of predictions with confidence scores.
        img_ids: List of image IDs to compute scores for.
        method: Uncertainty method - 'exponential', 'entropy', or 'margin'.

    Returns:
        Dict mapping image_id to uncertainty score.
    """
    scores = {}

    for img_id in img_ids:
        img_preds = predictions.get(str(img_id), [])

        if not img_preds:
            # No predictions means high uncertainty
            scores[img_id] = 1.0
            continue

        confidences = [p.get("confidence", p.get("score", 0.5)) for p in img_preds]

        if method == "exponential":
            # Exponential decay from confidence - lower confidence = higher uncertainty
            mean_conf = np.mean(confidences) if confidences else 0.5
            scores[img_id] = np.exp(-2 * mean_conf)
        elif method == "entropy":
            # Entropy-based uncertainty
            probs = np.clip(confidences, 1e-7, 1 - 1e-7)
            entropy = -np.mean(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
            scores[img_id] = entropy
        elif method == "margin":
            # Margin-based uncertainty (how close to decision boundary)
            margins = [abs(c - 0.5) for c in confidences]
            scores[img_id] = 1.0 - np.mean(margins) * 2 if margins else 1.0
        else:
            scores[img_id] = 1.0 - np.mean(confidences) if confidences else 1.0

    return scores


def compute_diversity_scores(
    features: np.ndarray,
    feature_img_ids: List[int],
    selected_ids: List[int],
    available_ids: List[int],
    distance: str = "euclidean",
    device: str = "cuda",
) -> dict:
    """
    Compute diversity scores based on feature space distances.

    Args:
        features: Feature array of shape [N, D].
        feature_img_ids: Image IDs corresponding to each feature row.
        selected_ids: Already selected image IDs.
        available_ids: Available image IDs for selection.
        distance: Distance metric - 'euclidean' or 'cosine'.
        device: Torch device for computation.

    Returns:
        Dict mapping image_id to diversity score.
    """
    if features is None or len(features) == 0:
        return {img_id: 1.0 for img_id in available_ids}

    # Create mapping from img_id to feature index
    id_to_idx = {img_id: idx for idx, img_id in enumerate(feature_img_ids)}

    # Get features for selected samples
    selected_indices = [id_to_idx[sid] for sid in selected_ids if sid in id_to_idx]

    if not selected_indices:
        return {img_id: 1.0 for img_id in available_ids}

    features_tensor = torch.from_numpy(features).float()
    if device == "cuda" and torch.cuda.is_available():
        features_tensor = features_tensor.cuda()

    selected_features = features_tensor[selected_indices]

    scores = {}
    for img_id in available_ids:
        if img_id not in id_to_idx:
            scores[img_id] = 1.0
            continue

        idx = id_to_idx[img_id]
        sample_feature = features_tensor[idx].unsqueeze(0)

        if distance == "euclidean":
            dists = torch.norm(selected_features - sample_feature, dim=1)
            min_dist = dists.min().item()
            # Normalize to [0, 1] range approximately
            scores[img_id] = min(min_dist / 10.0, 1.0)
        elif distance == "cosine":
            # Cosine distance
            sample_norm = sample_feature / (sample_feature.norm() + 1e-8)
            selected_norm = selected_features / (selected_features.norm(dim=1, keepdim=True) + 1e-8)
            similarities = (selected_norm @ sample_norm.T).squeeze()
            min_sim = similarities.max().item()
            scores[img_id] = 1.0 - min_sim
        else:
            scores[img_id] = 1.0

    return scores


def sample_FAST_images(
    selected_ids: List[int],
    n_images: int,
    last_exp_path: str,
    oracle_coco: COCO,
    masks: torch.Tensor,
    masks_gt: torch.Tensor,
    sam_boxes: np.ndarray,
    sam_idx: np.ndarray,
    agg_method: str = "max",
    save_memory: bool = True,
    device: str = "cuda",
    distance: str = "euclidean",
    uncertainty_method: str = "exponential",
    classifier=None,
    uncertainty_only: bool = False,
    alpha: float = 1.0,
    beta: float = 0.5,
    tintra: float = 0.7,
    tinter: float = 0.3,
    expand_ratio: float = 4.0,
    diversity_method: str = "original",
    img_size: int = 1024,
    base_coco_json: str = None,
) -> List[int]:
    """
    Sample images using the FAST active learning method.

    FAST combines uncertainty-based and diversity-based sampling:
    score = alpha * uncertainty + beta * diversity

    Args:
        selected_ids: Already labeled image IDs.
        n_images: Number of images to select.
        last_exp_path: Path to experiment directory with predictions.
        oracle_coco: COCO object with ground truth annotations.
        masks: SAM masks tensor.
        masks_gt: Ground truth masks tensor.
        sam_boxes: SAM predicted bounding boxes.
        sam_idx: Image indices for SAM boxes.
        agg_method: Aggregation method ('max', 'mean').
        save_memory: Whether to use memory-efficient computation.
        device: Torch device.
        distance: Distance metric for diversity.
        uncertainty_method: Method for uncertainty estimation.
        classifier: Optional classifier for uncertainty.
        uncertainty_only: If True, only use uncertainty (no diversity).
        alpha: Weight for uncertainty score.
        beta: Weight for diversity score.
        tintra: Intra-class threshold.
        tinter: Inter-class threshold.
        expand_ratio: Box expansion ratio.
        diversity_method: Method for diversity computation.
        img_size: Image size.
        base_coco_json: Path to base COCO JSON.

    Returns:
        List of selected image IDs.
    """
    # Get all image IDs from the oracle
    all_img_ids = oracle_coco.getImgIds()
    available_ids = list(set(all_img_ids) - set(selected_ids))

    if len(available_ids) <= n_images:
        return available_ids

    # Load predictions
    predictions_file = os.path.join(last_exp_path, "predictions.json")
    if os.path.exists(predictions_file):
        with open(predictions_file, "r") as f:
            predictions = json.load(f)
    else:
        predictions = {}

    # Compute uncertainty scores
    uncertainty_scores = compute_uncertainty_scores(
        predictions, available_ids, method=uncertainty_method
    )

    if uncertainty_only:
        # Sort by uncertainty and select top n_images
        sorted_ids = sorted(
            available_ids,
            key=lambda x: uncertainty_scores.get(x, 0.0),
            reverse=True,
        )
        return sorted_ids[:n_images]

    # Load features for diversity
    features_file = os.path.join(last_exp_path, "features_256_2.npy")
    if os.path.exists(features_file):
        features = np.load(features_file)
        # Assume features are ordered by image ID
        feature_img_ids = sorted(all_img_ids)
    else:
        features = None
        feature_img_ids = []

    # Compute diversity scores
    diversity_scores = compute_diversity_scores(
        features,
        feature_img_ids,
        selected_ids,
        available_ids,
        distance=distance,
        device=device,
    )

    # Combine scores
    combined_scores = {}
    for img_id in available_ids:
        unc = uncertainty_scores.get(img_id, 0.5)
        div = diversity_scores.get(img_id, 0.5)
        combined_scores[img_id] = alpha * unc + beta * div

    # Greedy selection with diversity update
    selected = []
    remaining = set(available_ids)

    for _ in range(n_images):
        if not remaining:
            break

        # Select image with highest combined score
        best_id = max(remaining, key=lambda x: combined_scores.get(x, 0.0))
        selected.append(best_id)
        remaining.remove(best_id)

        # Update diversity scores for remaining images
        if features is not None and not uncertainty_only:
            new_diversity = compute_diversity_scores(
                features,
                feature_img_ids,
                selected_ids + selected,
                list(remaining),
                distance=distance,
                device=device,
            )
            for img_id in remaining:
                unc = uncertainty_scores.get(img_id, 0.5)
                div = new_diversity.get(img_id, 0.5)
                combined_scores[img_id] = alpha * unc + beta * div

    return selected
