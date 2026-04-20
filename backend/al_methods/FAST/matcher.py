import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between predicted bounding boxes (with confidence scores)
    and target boxes based on a combination of L1 distance, negative generalized IoU,
    and a confidence-based multiplicative factor.
    """

    def __init__(self, weight_dict):
        """
        Args:
            weight_dict (dict): Dictionary with the following keys:
                - "cost_bbox": weight for the L1 cost between boxes.
                - "cost_giou": weight for the generalized IoU cost.
            cost_conf (float): Weight for the confidence-based factor.
                               (Larger values give lower–confidence predictions a higher cost.)
        """
        super().__init__()
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.cost_conf = weight_dict["cost_conf"]

        assert (
            self.cost_bbox != 0 or self.cost_giou != 0 or self.cost_conf != 0
        ), "All cost weights can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching.

        Args:
            outputs (dict): Must contain:
                - "pred_boxes": Tensor of shape [num_queries, 4] in cxcywh format.
                - "pred_scores": Tensor of shape [num_queries] with confidence scores.
            targets (dict): Must contain:
                - "boxes": Tensor of shape [num_target_boxes, 4] in cxcywh format.

        Returns:
            A tuple containing:
                - cost_matrix_np: numpy array representing the cost matrix.
                - giou_matrix_np: numpy array of generalized IoU values for each prediction-target pair.
                - indices: A list (length = batch_size) of tuples (index_pred, index_target)
                  (currently unused in this snippet).
        """
        pred_boxes = outputs["pred_boxes"]  # [num_queries, 4]
        pred_scores = outputs["pred_scores"]  # [num_queries]

        indices = []
        tgt_boxes = targets["boxes"]  # [num_targets, 4]

        # Compute L1 cost between boxes.
        cost_bbox = torch.cdist(
            pred_boxes, tgt_boxes, p=1
        )  # [num_queries, num_targets]
        avg_h, avg_w = tgt_boxes[:, 3].mean(), tgt_boxes[:, 2].mean()
        cost_bbox = cost_bbox / (avg_h + avg_w)

        # Compute generalized IoU.
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        # Compute the actual giou values.
        giou_matrix = generalized_box_iou(
            pred_boxes_xyxy, tgt_boxes_xyxy
        )  # [num_queries, num_targets]
        cost_giou = 1 - giou_matrix

        # Combined cost (without confidence factor).
        base_cost = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        cost_matrix = base_cost  # (the confidence factor is currently commented out)

        cost_matrix_np = cost_matrix.cpu().detach().numpy()
        giou_matrix_np = giou_matrix.cpu().detach().numpy()

        # (Linear sum assignment solving can go here)
        return cost_matrix_np, giou_matrix_np, indices

    @staticmethod
    def compute_uncertainty(
        cost_matrix_np, pred_scores, giou_matrix_np=None, confidence_threshold=0.8
    ):
        """
        Compute an uncertainty score from the cost matrix and predicted confidence scores.
        If the best matching giou for a prediction is negative, the uncertainty for that prediction is set to 0.

        Args:
            cost_matrix_np (np.ndarray): The cost matrix, shape [num_predictions, num_targets].
            pred_scores (torch.Tensor): Confidence scores for each prediction, shape [num_predictions].
            giou_matrix_np (np.ndarray, optional): Generalized IoU values, same shape as cost_matrix_np.
            confidence_threshold (float): Only predictions with confidence below this threshold are considered uncertain.

        Returns:
            uncertainty (float): An aggregated uncertainty score.
        """
        # For each predicted box, get the minimum cost (best match) across targets.
        best_match_cost = cost_matrix_np.min(axis=1)  # [num_predictions]
        best_match_indices = cost_matrix_np.argmin(
            axis=1
        )  # corresponding target indices

        # If a giou matrix is provided, check if the best match has negative giou.
        if giou_matrix_np is not None:
            best_giou = giou_matrix_np[
                np.arange(len(best_match_indices)), best_match_indices
            ]
            # For predictions with negative giou (i.e. no overlap), set cost to 0.
            best_match_cost[best_giou < 0] = 0.0

        # Convert pred_scores to numpy array if necessary.
        pred_scores_np = (
            pred_scores.cpu().detach().numpy()
            if torch.is_tensor(pred_scores)
            else pred_scores
        )

        # Only consider predictions with confidence below the threshold.
        uncertain_mask = pred_scores_np < confidence_threshold
        if np.sum(uncertain_mask) == 0:
            return 0.0

        uncertainty = best_match_cost[uncertain_mask].mean()
        return uncertainty

    @staticmethod
    def compute_uncertainty_intelligent(
        cost_matrix_np,
        pred_scores,
        uncertainty_method="original",
        giou_matrix_np=None,
        **kwargs
    ):
        """
        Compute an uncertainty score from the cost matrix and predicted confidence scores,
        using an "intelligent" formulation.

        If the best matching giou for a prediction is negative, the uncertainty for that prediction is set to 0.

        The procedure is:
          1. For each prediction, select the best matching cost across targets.
          2. Convert this cost to a matching quality score m via an exponential: m = exp(-cost).
          3. Combine m with the predicted confidence c (in [0,1]) to form U(c, m).
          4. Aggregate the per-prediction uncertainties.

        Args:
            cost_matrix_np (np.ndarray): Cost matrix of shape [num_predictions, num_targets].
            pred_scores (torch.Tensor or np.ndarray): Confidence scores for each prediction.
            giou_matrix_np (np.ndarray, optional): Generalized IoU values, used to zero-out uncertainty when negative.
            alpha, beta, gamma, delta, lam: Hyperparameters for different methods.

        Returns:
            aggregated_uncertainty (float): Aggregated uncertainty score.
            U (np.ndarray): Per-prediction uncertainties.
        """
        if cost_matrix_np.size == 0:
            return 0.0, np.array([]), (np.array([]), np.array([]))
        best_match_cost = cost_matrix_np.min(axis=1)
        best_match_indices = cost_matrix_np.argmin(axis=1)

        # Create a mask for predictions whose best match has a negative giou.
        if giou_matrix_np is not None:
            best_giou = giou_matrix_np[
                np.arange(len(best_match_indices)), best_match_indices
            ]
            negative_giou_mask = best_giou < 0
        else:
            negative_giou_mask = np.zeros_like(best_match_cost, dtype=bool)

        # Compute matching quality from cost.
        m = np.exp(-best_match_cost)
        c = (
            pred_scores.cpu().detach().numpy()
            if torch.is_tensor(pred_scores)
            else pred_scores
        )

        alpha = kwargs.get("alpha", 1.0)
        beta = kwargs.get("beta", 1.0)
        gamma = kwargs.get("gamma", 1.0)
        delta = kwargs.get("delta", 1.0)
        lam = kwargs.get("lam", 5.0)

        if uncertainty_method == "original":
            U = alpha * np.abs(c - m) + beta * (1 - c) * (1 - m)
        elif uncertainty_method == "original_wo":
            U = alpha * np.abs(c - m)
        elif uncertainty_method == "product":
            U = 1 - c * m
        elif uncertainty_method == "harmonic":
            epsilon = 1e-6
            U = 1 - (2 * c * m) / (c + m + epsilon)
        elif uncertainty_method == "combined":
            U = gamma * np.abs(c - m) + delta * (1 - (c + m) / 2)
        elif uncertainty_method == "exponential":
            U = np.exp(-lam * np.minimum(c, m))
        else:
            raise ValueError(
                "Unknown uncertainty method: {}".format(uncertainty_method)
            )

        # Set uncertainty to 0 for predictions where the best matching giou is negative.
        U[negative_giou_mask] = 0.0

        aggregated_uncertainty = U.mean() if U.size > 0 else 0.0
        indices = (best_match_indices, np.arange(len(best_match_indices)))
        return aggregated_uncertainty, U, indices
