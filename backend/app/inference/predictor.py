"""ONNX Runtime image classifier used by the API (no PyTorch at inference time)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


class Predictor:
    def __init__(self, model_path: str, threshold: float = 0.60, image_size: int = 224):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.image_size = image_size
        self.session = None
        self.classes: list[str] = []
        self.architecture = "MobileNetV3-Small"
        self.device = "CPUExecutionProvider"
        metadata_path = self.model_path.with_suffix(".json")

        if self.model_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.classes = metadata["classes"]
            self.image_size = int(metadata.get("image_size", image_size))
            self.threshold = threshold
            self.architecture = metadata.get("architecture", self.architecture)
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required to load the ONNX model; install backend/requirements.txt") from exc
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )

    @property
    def ready(self) -> bool:
        return self.session is not None

    @staticmethod
    def preprocess(image: Image.Image, image_size: int = 224) -> np.ndarray:
        """Resize/crop RGB image and apply ImageNet channel normalization."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if not width or not height or max(width, height) > 8192:
            raise ValueError("Invalid or excessively large image dimensions")

        resize_size = int(round(image_size * 256 / 224))
        scale = resize_size / min(width, height)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        left = (image.width - image_size) // 2
        top = (image.height - image_size) // 2
        image = image.crop((left, top, left + image_size, top + image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
        return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)

    def predict(self, image: Image.Image) -> dict:
        if not self.ready:
            return {
                "status": "model_unavailable",
                "message": "Train/export a model and set MODEL_PATH to its ONNX file.",
            }
        tensor = self.preprocess(image, self.image_size)
        input_name = self.session.get_inputs()[0].name  # type: ignore[union-attr]
        output_name = self.session.get_outputs()[0].name  # type: ignore[union-attr]
        logits = self.session.run([output_name], {input_name: tensor})[0][0]
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.exp(logits).sum()
        top_indices = np.argsort(probabilities)[::-1][: min(3, len(self.classes))]
        alternatives = [
            {"label": self.classes[int(i)], "confidence": float(probabilities[i])}
            for i in top_indices
        ]
        best = alternatives[0]
        if best["confidence"] < self.threshold:
            return {
                "status": "unknown",
                "label": "Unknown sign",
                "confidence": best["confidence"],
                "alternatives": alternatives,
            }
        return {
            "status": "ok",
            "label": best["label"],
            "confidence": best["confidence"],
            "alternatives": alternatives,
        }
