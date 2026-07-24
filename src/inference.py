"""Inference utilities for single-image prediction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from src.data.dataset import build_normalize_fn
from src.data.preprocessing import PreprocessingPipeline
from src.gradcam import GradCAM
from src.models.classifier import build_model
from src.report import build_medical_report, encode_image_base64
from src.train import resolve_device
from src.utils.config import AppConfig


class BrainTumorPredictor:
    """Load a trained checkpoint and run inference on MRI images."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        checkpoint_path = Path(checkpoint_path)
        device_obj = resolve_device(device or "cuda")
        payload = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        self.config: AppConfig = payload["config"]
        self.device = device_obj
        self.model = build_model(self.config.model, num_classes=self.config.data.num_classes)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.preprocess = PreprocessingPipeline(
            self.config.preprocessing,
            self.config.data.image_size,
        )
        self.normalize_fn = build_normalize_fn(self.config)
        self.class_names = self.config.data.class_names
        self._gradcam: GradCAM | None = None

    @property
    def gradcam(self) -> GradCAM:
        """Lazy-init Grad-CAM helper."""
        if self._gradcam is None:
            self._gradcam = GradCAM(self.model)
        return self._gradcam

    def _prepare_tensor(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], torch.Tensor]:
        """Run preprocessing and return processed image + model input tensor."""
        processed = self.preprocess(image)
        tensor = self.normalize_fn(processed).unsqueeze(0).to(self.device)
        return processed, tensor

    def predict_from_array(self, image: NDArray[np.uint8]) -> dict:
        """Predict class and confidence from a BGR uint8 image array."""
        _, tensor = self._prepare_tensor(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        return {
            "class_name": self.class_names[pred_idx],
            "class_index": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            },
        }

    def analyze_from_array(
        self,
        image: NDArray[np.uint8],
        patient_id: str | None = None,
    ) -> dict:
        """Full analysis: prediction, Grad-CAM overlay, and medical report."""
        processed, tensor = self._prepare_tensor(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        class_name = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        probabilities = {name: float(probs[i]) for i, name in enumerate(self.class_names)}

        cam = self.gradcam.generate(tensor, target_class=pred_idx)
        gradcam_overlay = self.gradcam.overlay_on_image(processed, cam)

        report = build_medical_report(
            class_name=class_name,
            confidence=confidence,
            probabilities=probabilities,
            original_image=image,
            patient_id=patient_id,
        )

        return {
            "class_name": class_name,
            "class_index": pred_idx,
            "confidence": confidence,
            "probabilities": probabilities,
            "gradcam_image": encode_image_base64(gradcam_overlay),
            "original_image": encode_image_base64(processed),
            "report": report,
        }

    def predict_from_path(self, image_path: str | Path) -> dict:
        """Predict from an image file path."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        result = self.predict_from_array(image)
        result["image_path"] = str(image_path)
        return result
