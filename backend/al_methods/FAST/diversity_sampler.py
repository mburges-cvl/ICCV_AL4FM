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


def compute_prototypes(
    agg,
    available_img_ids,
    coco,
    features,
    masks,
    masks_gt,
    clustering=False,
    n_clusters=5,
    device="cuda",
    save_memory=False,
):
    """
    Compute per-class prototypes (or cluster centers). Any instance mask with no overlap
    with the annotation's bounding box is labeled as class 0 (background).
    """

    all_gt_features = []
    all_gt_classes = []

    selected_img_ids = list(available_img_ids)
    # if labeled_examples > 0 and labeled_examples < len(selected_img_ids):
    #     np.random.shuffle(selected_img_ids)
    #     selected_img_ids = selected_img_ids[:labeled_examples]

    imgid_to_idx = {img_id: idx for idx, img_id in enumerate(available_img_ids)}

    for img_id in tqdm(selected_img_ids, total=len(selected_img_ids)):
        img_info = coco.loadImgs([img_id])[0]
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if img_id not in imgid_to_idx:
            print(f"Warning: image ID {img_id} not found in available_img_ids!")
            continue
        idx = imgid_to_idx[img_id]

        masks_as_one = masks[idx].copy()
        masks_gt_as_one = masks_gt[idx].copy()
        features_as_one = features[idx]

        masks_as_one = torch.from_numpy(masks_as_one).long()
        masks_gt_as_one = torch.from_numpy(masks_gt_as_one).long()

        if save_memory:
            masks_as_one = masks_as_one.to(device)
            masks_gt_as_one = masks_gt_as_one.to(device)
            features_as_one = features_as_one.to(device)

        H, W = masks_gt_as_one.shape

        sample_features, sample_features_gt = get_per_mask_features(
            features_as_one, masks_as_one, agg, mask_gt=masks_gt_as_one, device=device
        )

        # Positive examples from ground truth aggregation.
        for ann, feat in zip(annotations, sample_features_gt):
            all_gt_features.append(feat)
            all_gt_classes.append(ann["category_id"])

        # Remove background (index 0) from instance features.
        inst_features = sample_features[1:]
        if inst_features.shape[0] == 0:
            continue

        # For each annotation, add negative examples (class 0) when there is no overlap.
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            x1 = max(0, int(math.floor(x)))
            y1 = max(0, int(math.floor(y)))
            x2 = min(W, x1 + int(math.ceil(w)))
            y2 = min(H, y1 + int(math.ceil(h)))
            if x2 <= x1 or y2 <= y1:
                continue
            sub_mask = masks_as_one[y1:y2, x1:x2]
            if sub_mask.numel() == 0:
                continue
            # Count pixel occurrences per instance using torch.bincount.
            n_instances = inst_features.shape[0]
            counts = torch.bincount(sub_mask.flatten(), minlength=n_instances + 1)[1:]
            no_overlap = counts == 0
            if no_overlap.any():
                neg_feats = inst_features[no_overlap]
                # Append all negative features with class 0.
                for feat in neg_feats:
                    all_gt_features.append(feat)
                    all_gt_classes.append(0)

    all_gt_features = torch.stack(all_gt_features)
    all_gt_classes = torch.tensor(all_gt_classes, device=all_gt_features.device)
    unique_classes = torch.unique(all_gt_classes)

    prototypes = {}
    for cls in unique_classes:
        class_feats = all_gt_features[all_gt_classes == cls]
        if class_feats.shape[0] == 0:
            continue
        if clustering:
            centers = kmeans_torch(class_feats, n_clusters)
            prototypes[int(cls.item())] = centers.half()
        else:
            prototypes[int(cls.item())] = class_feats.mean(dim=0).half()

    if save_memory:
        for k, v in prototypes.items():
            prototypes[k] = v.cpu()

    return prototypes, selected_img_ids


def kmeans_torch(X, n_clusters, n_iters=100):
    if X.size(0) < n_clusters:
        indices = torch.randint(0, X.size(0), (n_clusters,))
    else:
        indices = torch.randperm(X.size(0))[:n_clusters]
    centers = X[indices].clone()
    # X: (N, feature_dim)
    centers = X[indices].clone()
    for _ in range(n_iters):
        distances = torch.cdist(X, centers, p=2)
        labels = torch.argmin(distances, dim=1)
        new_centers = []
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if cluster_points.shape[0] == 0:
                new_centers.append(centers[k])
            else:
                new_centers.append(cluster_points.mean(dim=0))
        new_centers = torch.stack(new_centers)
        if torch.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    return centers


def get_per_mask_features(features, mask, agg, mask_gt=None, device="cuda"):

    sample_features_gt = None

    H, W = mask.shape
    features_mask_sized = F.interpolate(
        features.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    # shape (C, H, W)

    # Flatten masks and features
    mask_flat = mask.view(-1).long()
    feat_flat = features_mask_sized.view(features_mask_sized.shape[0], -1).permute(1, 0)

    # Aggregate features by instance
    if agg == "mean":
        # scatter_mean returns (#instances+1, C)
        sample_features = torch_scatter.scatter_mean(feat_flat, mask_flat, dim=0)
        if mask_gt is not None:
            sample_features_gt = torch_scatter.scatter_mean(
                feat_flat, mask_gt.view(-1).long(), dim=0
            )
    elif agg == "max":
        # scatter_max returns (values, indices)
        sample_features, _ = torch_scatter.scatter_max(feat_flat, mask_flat, dim=0)
        if mask_gt is not None:
            sample_features_gt, _ = torch_scatter.scatter_max(
                feat_flat, mask_gt.view(-1).long(), dim=0
            )
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")

    # sample_features = sample_features.cpu().numpy()  # shape (num_instances+1, C)
    # if mask_gt is not None:
    #     sample_features_gt = sample_features_gt.cpu().numpy()

    return sample_features, sample_features_gt


def compute_image_diversity_score(
    prototypes,
    features,
    mask,
    agg,
    device="cuda",
    match_indices=None,
    distance="cosine",
):
    """
    Compute a diversity score for a single image based on the diversity of its
    matched instance (mask) features. Instead of filtering out background or low-confidence
    detections, only the masks indicated by match_indices (i.e. detections) are considered.

    Steps:
      1. Compute per-mask features.
      2. If match_indices is provided, select only those features.
      3. For each selected feature, compute the distance to each prototype using the specified distance metric.
      4. Assign the mask the label of the closest prototype and compute the margin between the two smallest distances.
      5. Group by predicted class and compute intra-class diversity (max pairwise distance).
      6. Return the mean of these diversities.
    """
    # Obtain per-mask features (features_per_mask shape: (num_masks, C))
    mask_tensor = torch.from_numpy(mask).long().to(device)
    features_per_mask, _ = get_per_mask_features(
        features, mask_tensor, agg, device=device
    )

    # Only keep the matched masks as provided by the uncertainty sampler.
    if match_indices is not None:

        # Assumes match_indices is a list or tensor of indices corresponding to the valid detections.
        features_per_mask = features_per_mask[match_indices[0], :]

    # If no detections remain, return a diversity score of 0.
    if features_per_mask.shape[0] == 0:
        return 0

    # Use all detections (no removal for background)
    proto_items = sorted(prototypes.items(), key=lambda x: x[0])
    predicted_classes = []
    margins = []  # margin computed per detection

    for feat in features_per_mask:
        feat = feat.unsqueeze(0)  # shape (1, C)
        distances = []
        for cls, proto in proto_items:
            proto = proto.float().to(device)
            if distance == "cosine":
                if proto.ndim == 2:
                    # When multiple prototypes per class exist (from clustering)
                    sim = F.cosine_similarity(feat, proto, dim=1)
                    d = 1 - sim  # cosine distance
                    d_min = d.min().item()
                else:
                    sim = F.cosine_similarity(feat, proto.unsqueeze(0), dim=1)
                    d_min = (1 - sim).item()
            elif distance == "euclidean":
                if proto.ndim == 2:
                    d = torch.cdist(feat, proto, p=2)
                    d_min = d.min().item()
                else:
                    d = torch.cdist(feat, proto.unsqueeze(0), p=2)
                    d_min = d.min().item()
            else:
                raise ValueError(f"Unknown distance metric: {distance}")
            distances.append(d_min)
        distances_tensor = torch.tensor(distances, device=device)
        sorted_distances, _ = torch.sort(distances_tensor)
        margin = (
            sorted_distances[1] - sorted_distances[0]
            if len(sorted_distances) > 1
            else sorted_distances[0]
        )
        min_val, min_idx = torch.min(distances_tensor, dim=0)
        predicted_class = proto_items[min_idx][0]
        predicted_classes.append(predicted_class)
        margins.append(margin.cpu())

    # Group the features by their predicted class
    grouped_features = {}
    for i, cls in enumerate(predicted_classes):
        if cls not in grouped_features:
            grouped_features[cls] = []
        grouped_features[cls].append(features_per_mask[i])

    # For each predicted class, compute the intra-class diversity
    intra_class_diversities = []
    for cls, feats in grouped_features.items():
        feats_tensor = torch.stack(feats)  # shape (n_detections, C)
        if feats_tensor.shape[0] == 1:
            intra_diversity = 0  # with one detection, diversity is zero
        else:
            # Compute all pairwise distances and take the maximum as the diversity measure.
            dists = torch.cdist(feats_tensor, feats_tensor, p=2)
            intra_diversity = dists.max().item()
        intra_class_diversities.append(intra_diversity)

    # diversity_score = np.mean(intra_class_diversities) if intra_class_diversities else 0

    average_margin = np.mean(margins) if margins else 0
    # Combine both signals (you can adjust the weighting)
    diversity_score = (
        np.mean(intra_class_diversities) if intra_class_diversities else 0
    ) * (1 - average_margin)
    return diversity_score


def get_diverse_images_by_mask_diversity(
    coco,
    prototypes,
    image_ids_pool,
    features,
    masks,
    agg,
    distance,
    n_images=0,
    device="cuda",
    match_indices=None,
):
    """
    For each candidate image in image_ids_pool, compute a diversity score based on
    the diversity of the matched instance (mask) features.

    If match_indices is provided, it is assumed to be a list (or similar structure)
    where each element corresponds to the indices of the matched detections for that image.
    """
    all_img_ids = coco.getImgIds()
    diversity_scores = np.zeros(len(image_ids_pool))

    for i, img_id in tqdm(enumerate(image_ids_pool), total=len(image_ids_pool)):
        try:
            mask_idx = all_img_ids.index(img_id)
        except ValueError:
            diversity_scores[i] = 0
            continue

        features_ = features[mask_idx].to(device)
        mask_ = masks[mask_idx].copy()

        # If match_indices are provided, extract the ones for the current image.
        current_match_indices = match_indices[i] if match_indices is not None else None

        score = compute_image_diversity_score(
            prototypes,
            features_,
            mask_,
            agg,
            device=device,
            match_indices=current_match_indices,
            distance=distance,
        )
        diversity_scores[i] = score

    # Sort images by diversity score (higher is better) and select the top n_images.
    sorted_indices = np.argsort(diversity_scores)
    sorted_img_ids = np.array(image_ids_pool)[sorted_indices]
    selected_img_ids = sorted_img_ids[-n_images:]  # take the top n_images

    return selected_img_ids.tolist()
