import torch
from torchvision.ops import nms
import numpy as np
import json
import os


def apply_nms_torch(boxes, scores, iou_threshold=0.3):
    boxes_xyxy = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]
    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_tensor, scores_tensor, iou_threshold)
    return keep.numpy().tolist()


def compute_giou(boxA, boxB):
    boxA_xyxy = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB_xyxy = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
    x1 = max(boxA_xyxy[0], boxB_xyxy[0])
    y1 = max(boxA_xyxy[1], boxB_xyxy[1])
    x2 = min(boxA_xyxy[2], boxB_xyxy[2])
    y2 = min(boxA_xyxy[3], boxB_xyxy[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = (boxA_xyxy[2] - boxA_xyxy[0]) * (boxA_xyxy[3] - boxA_xyxy[1])
    areaB = (boxB_xyxy[2] - boxB_xyxy[0]) * (boxB_xyxy[3] - boxB_xyxy[1])
    union_area = areaA + areaB - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    cx1 = min(boxA_xyxy[0], boxB_xyxy[0])
    cy1 = min(boxA_xyxy[1], boxB_xyxy[1])
    cx2 = max(boxA_xyxy[2], boxB_xyxy[2])
    cy2 = max(boxA_xyxy[3], boxB_xyxy[3])
    convex_area = (cx2 - cx1) * (cy2 - cy1)
    giou = iou - ((convex_area - union_area) / convex_area) if convex_area > 0 else iou
    return giou


def load_annotations_for_ids(
    predictions,
    new_img_ids,
    sam_boxes=None,
    sam_idx=None,
    nms_iou_threshold=0.5,
    use_sam_masks=False,
):
    with open(predictions, "r") as f:
        predictions_data = json.load(f)
    certain_boxes = {}
    for img_id in new_img_ids:
        pred = predictions_data.get(str(img_id), {})
        img_id = int(img_id)
        labels = pred.get("labels", [])
        boxes = pred.get("boxes", [])
        boxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes]
        scores = pred.get("scores", [])
        threshold = 0.01
        filtered_indices = [i for i, score in enumerate(scores) if score > threshold]
        if not filtered_indices:
            continue
        filtered_boxes = [boxes[i] for i in filtered_indices]
        filtered_scores = [scores[i] for i in filtered_indices]
        filtered_scores = np.array(filtered_scores)
        filtered_scores = (filtered_scores - np.min(filtered_scores)) / (
            np.max(filtered_scores) - np.min(filtered_scores) + 1e-6
        )
        filtered_scores = filtered_scores.tolist()
        filtered_labels = [labels[i] for i in filtered_indices]
        keep_indices = apply_nms_torch(
            filtered_boxes, filtered_scores, iou_threshold=nms_iou_threshold
        )
        pred_nms_boxes = [filtered_boxes[idx] for idx in keep_indices]
        sam_match_mapping = {}
        sam_boxes_image = None
        if sam_boxes is not None and sam_idx is not None and use_sam_masks:
            sam_boxes_idx = (sam_idx == img_id).squeeze()
            sam_boxes_image = sam_boxes[sam_boxes_idx]
            if len(sam_boxes_image) > 0:
                matches = []
                for i, pred_box in enumerate(pred_nms_boxes):
                    for j, sam_box in enumerate(sam_boxes_image):
                        giou_val = compute_giou(pred_box, sam_box)
                        if giou_val > 0.5:
                            matches.append((i, j, giou_val))
                matches.sort(key=lambda x: x[2], reverse=True)
                used_pred = set()
                used_sam = set()
                for i, j, giou_val in matches:
                    if i not in used_pred and j not in used_sam:
                        sam_match_mapping[i] = j
                        used_pred.add(i)
                        used_sam.add(j)
        certain_boxes[img_id] = []
        for i, idx in enumerate(keep_indices):
            if i in sam_match_mapping and sam_boxes_image is not None:
                replacement_box = sam_boxes_image[sam_match_mapping[i]]
                if isinstance(replacement_box, torch.Tensor):
                    box = replacement_box.tolist()
                elif isinstance(replacement_box, np.ndarray):
                    box = replacement_box.tolist()
                else:
                    box = replacement_box
            else:
                box = filtered_boxes[idx]
            score = filtered_scores[idx]
            certain_boxes[img_id].append(
                {
                    "bbox": box,
                    "confidence": score,
                    "group_id": int(filtered_labels[idx]),
                }
            )
    new_img_ids = [int(id) for id in new_img_ids]
    return {"new_training_ids": new_img_ids, "certain_boxes": certain_boxes}
