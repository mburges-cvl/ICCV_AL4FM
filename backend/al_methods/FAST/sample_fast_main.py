import os
import json
import logging
import numpy as np
import torch
from scipy.spatial.distance import cdist

from ..DivProto.divproto_sampler import (
    get_diverse_images_divproto,
    get_uncertain_images_divproto,
)
from ..PPAL import DiversitySampler, get_img_score_distance_matrix_slow, get_inter_feats
from ..CoreSet.sample_coreset_main import k_centroid_greedy

from .uncertainty_sampler import get_uncertain_images_box
from .diversity_sampler import (
    compute_prototypes,
    get_diverse_images_by_mask_diversity,
)

def sample_FAST_images(
    selected_ids,
    n_images,
    last_exp_path,
    oracle_coco,
    masks,
    masks_gt,
    sam_boxes,
    sam_idx,
    agg_method,
    save_memory,
    device,
    distance,
    uncertainty_method,
    classifier,
    uncertainty_only,
    alpha,
    beta,
    tintra,  # threshold for intra-class diversity (ENMS/DivProto)
    tinter,  # threshold for inter-class diversity (ENMS/DivProto)
    expand_ratio,
    diversity_method,
    img_size,
    base_coco_json,
):
    logging.info(f"Loading features from {last_exp_path}")
    feature_file = os.path.join(last_exp_path, "features_256_2.npy")
    predictions_file = os.path.join(last_exp_path, "predictions.json")

    # Load the sample predictions
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    if uncertainty_only:
        n_images_uc = n_images
    else:
        n_images_uc = n_images * expand_ratio

    # num_classes = len(oracle_coco.cats) + 1

    # Uncertainty sampling
    sampled_img_ids, match_indices = get_uncertain_images_box(
        n_images=n_images_uc,
        coco=oracle_coco,
        sam_boxes=sam_boxes,
        box_to_idx=sam_idx,
        selected_image_ids=selected_ids,
        predictions=predictions,
        alpha=alpha,
        beta=beta,
        uncertainty_method=uncertainty_method,
    )

    if uncertainty_only:
        os.remove(feature_file)
        return sampled_img_ids

    if diversity_method == "DivProto":
        features = np.load(feature_file, mmap_mode="r")

        all_img_ids = [str(img_id) for img_id in oracle_coco.getImgIds()]
        features_map = {img_id: features[i] for i, img_id in enumerate(all_img_ids)}
        unlabeled_img_ids = sampled_img_ids

        unlabeled_predictions = {
            img_id: predictions[img_id]
            for img_id in unlabeled_img_ids
            if img_id in predictions
        }
        unlabeled_features = {
            img_id: features_map[img_id]
            for img_id in unlabeled_img_ids
            if img_id in features_map
        }

        uncertain_images = get_uncertain_images_divproto(
            unlabeled_predictions, unlabeled_features, img_size=img_size
        )

        diverse_images = get_diverse_images_divproto(
            uncertain_images, n_images, tintra, tinter
        )

        sampled_diverse_img_ids = [int(img["id"]) for img in diverse_images]

    elif diversity_method == "PPAL":
        features = np.load(feature_file, mmap_mode="r")

        diversity_sampler = DiversitySampler(
            n_sample_images=10,
            oracle_annotation_path=base_coco_json,
        )

        queue_length = len(sampled_img_ids)
        max_det = 100
        feat_dim = features.shape[1]

        image_id_queue = torch.zeros((queue_length, 1), dtype=torch.int) - 1
        det_label_queue = torch.zeros((queue_length, max_det))
        det_score_queue = torch.zeros((queue_length, max_det))
        det_feat_queue = torch.zeros((queue_length, max_det, feat_dim))

        indx = 0
        for image_id, result in predictions.items():
            if int(image_id) not in sampled_img_ids:
                continue

            boxes = torch.tensor(result["boxes"])[:max_det]
            feature_id = result["image_idx"]
            features_ = torch.from_numpy(features[feature_id]).unsqueeze(0).float()
            lvl_idx = torch.zeros((boxes.shape[0]), dtype=torch.int)

            inter_feats = get_inter_feats(features_, lvl_idx, boxes, [1024, 1024, 3])

            image_id_queue[indx] = int(image_id)
            det_label_queue[indx, : boxes.shape[0]] = torch.tensor(result["labels"])[
                :max_det
            ].long()
            det_score_queue[indx, : boxes.shape[0]] = torch.tensor(result["scores"])[
                :max_det
            ].float()
            det_feat_queue[indx, : boxes.shape[0]] = inter_feats
            indx += 1

        img_dis_mat = (
            get_img_score_distance_matrix_slow(
                det_label_queue, det_score_queue, det_feat_queue, score_thr=0.005
            )
            .cpu()
            .numpy()
        )

        diversity_sampler.n_images = n_images
        sampled_diverse_img_ids = diversity_sampler.al_round(
            img_dis_mat,
            np.array(sampled_img_ids),
        )

    elif diversity_method == "CoreSet":
        features = np.load(feature_file)
        # Reshape features if needed.
        features = features.reshape(features.shape[0], features.shape[1], -1)

        available_ids = oracle_coco.getImgIds()

        # Get indices for each set of ids.
        sampled_idxs = [available_ids.index(img_id) for img_id in sampled_img_ids]
        selected_idxs = [available_ids.index(img_id) for img_id in selected_ids]

        # Combine indices: start with selected_ids to preserve their order,
        # then add any new indices from sampled_img_ids that aren't already in selected.
        combined_idxs = list(dict.fromkeys(selected_idxs + sampled_idxs))

        # Sub-select the features for both groups.
        features = features[combined_idxs]

        # Pool along the spatial dimensions.
        if agg_method == "max":
            features = features.max(axis=2)
        elif agg_method == "mean":
            features = features.mean(axis=2)

        # Compute the distance matrix.
        dis_matrix = cdist(features, features, metric="euclidean")

        # Map the combined indices back to the actual image ids.
        combined_ids = [available_ids[idx] for idx in combined_idxs]
        index_to_id = {i: img_id for i, img_id in enumerate(combined_ids)}

        # Determine the positions (in the combined features) corresponding to selected_ids.
        already_selected_indices = [
            combined_ids.index(img_id)
            for img_id in selected_ids
            if img_id in combined_ids
        ]

        # Run k_centroid_greedy using the combined distance matrix.
        centroids = k_centroid_greedy(
            dis_matrix, additional_K=n_images, already_selected=already_selected_indices
        )

        sampled_diverse_img_ids = [index_to_id[idx] for idx in centroids]

    elif diversity_method == "original":

        features = np.load(feature_file)
        features = torch.from_numpy(features).float()

        if not save_memory:
            features = features.to(device)

        # Compute prototypes from the labelled data (as before)
        diversity_prototypes, _ = compute_prototypes(
            agg=agg_method,
            available_img_ids=selected_ids,
            coco=oracle_coco,
            features=features,
            masks=masks,
            masks_gt=masks_gt,
            clustering=True,
            n_clusters=5,
            device=device,
            save_memory=save_memory,
        )

        # Now sample diverse images by computing per-image prototypes for the unlabelled images
        sampled_diverse_img_ids = get_diverse_images_by_mask_diversity(
            coco=oracle_coco,
            prototypes=diversity_prototypes,
            image_ids_pool=sampled_img_ids,
            features=features,
            masks=masks,
            agg=agg_method,
            distance=distance,
            n_images=n_images,
            device=device,
            # confidence_threshold=0.1,
            match_indices=match_indices,
        )
    else:
        raise ValueError(f"Unknown diversity method: {diversity_method}")
    # Optionally remove the feature file after sampling.
    if os.path.exists(feature_file):
        os.remove(feature_file)
        logging.info(f"Removed feature file {feature_file}")

    return sampled_diverse_img_ids
