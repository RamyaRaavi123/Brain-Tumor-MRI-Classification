"""Grad-CAM explainability for brain tumor MRI classification."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from src.models.classifier import BrainTumorClassifier


def _find_last_conv_layer(module: nn.Module) -> nn.Module:
    """Return the deepest Conv2d layer in a module tree."""
    last_conv: nn.Module | None = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model backbone.")
    return last_conv


class GradCAM:
    """Generate Grad-CAM heatmaps for BrainTumorClassifier predictions."""

    def __init__(self, model: BrainTumorClassifier) -> None:
        self.model = model
        self.model.eval()
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.target_layer = _find_last_conv_layer(model.backbone)
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self,
        _module: nn.Module,
        _inputs: tuple,
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> NDArray[np.float32]:
        """Compute normalized Grad-CAM heatmap for the given input."""
        self.model.zero_grad(set_to_none=True)
        self.gradients = None
        self.activations = None

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy().astype(np.float32)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_on_image(
        self,
        image: NDArray[np.uint8],
        cam: NDArray[np.float32],
        alpha: float = 0.45,
    ) -> NDArray[np.uint8]:
        """Blend heatmap onto a BGR uint8 image."""
        if image.ndim == 2:
            display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display = image.copy()

        h, w = display.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(display, 1 - alpha, heatmap, alpha, 0)
        return blended

    def close(self) -> None:
        """Remove registered hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()
