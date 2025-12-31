"""
Compatibility shim for samgeo.hq_sam.SamGeo

This provides a minimal implementation that wraps the standard SAM library
when the full samgeo package is not available.
"""

import numpy as np
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment_anything not available. SAM features will be disabled.")


class SamGeo:
    """
    Minimal SamGeo implementation for compatibility.

    This wraps the standard segment-anything library to provide
    the API expected by the server code.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: str = None,
        sam_kwargs: dict = None,
        automatic: bool = True,
        hq: bool = False,
        device: str = "cuda",
    ):
        self.model_type = model_type
        self.automatic = automatic
        self.hq = hq
        self.device = device if torch.cuda.is_available() else "cpu"
        self.sam_kwargs = sam_kwargs or {}
        self.masks = []
        self.mask_generator = None
        self.predictor = None

        if not SAM_AVAILABLE:
            print("Warning: SAM not available. Using mock implementation.")
            return

        # SAM checkpoint paths (users should download these)
        checkpoint_paths = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }

        if checkpoint is None:
            checkpoint = checkpoint_paths.get(model_type)

        # Try to load the model
        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
            self.sam.to(device=self.device)

            if automatic:
                self.mask_generator = SamAutomaticMaskGenerator(
                    self.sam,
                    points_per_side=self.sam_kwargs.get("points_per_side", 32),
                    points_per_batch=self.sam_kwargs.get("points_per_batch", 64),
                )
            else:
                self.predictor = SamPredictor(self.sam)
        except Exception as e:
            print(f"Warning: Could not load SAM model: {e}")
            print("SAM features will use mock implementations.")
            self.sam = None

    def set_image(self, image: np.ndarray):
        """Set the image for prediction."""
        if self.predictor is not None:
            self.predictor.set_image(image)
        self._current_image = image

    def generate(self, image: np.ndarray, foreground: bool = True, unique: bool = True):
        """Generate masks for the entire image."""
        self._current_image = image

        if self.mask_generator is not None:
            self.masks = self.mask_generator.generate(image)
        else:
            # Mock implementation - return empty masks
            h, w = image.shape[:2]
            self.masks = []
            print("Warning: Using mock mask generation - no actual masks generated.")

    def predict(self, boxes: list = None, return_results: bool = False):
        """Predict mask for given boxes."""
        if self.predictor is None or not hasattr(self, '_current_image'):
            # Mock implementation
            h, w = 1024, 1024
            if hasattr(self, '_current_image'):
                h, w = self._current_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    mask[y1:y2, x1:x2] = 1
            if return_results:
                return [mask], [1.0], None
            return mask

        # Use actual SAM predictor
        if boxes:
            box = np.array(boxes[0])
            masks, scores, logits = self.predictor.predict(
                box=box,
                multimask_output=False,
            )
            if return_results:
                return masks, scores, logits
            return masks[0]

        return None


# Alias for compatibility
HQSamGeo = SamGeo
