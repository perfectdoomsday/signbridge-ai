from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ml.models.image_classifier import build_image_model


def main():
    parser = argparse.ArgumentParser(description="Export a SignBridge PyTorch checkpoint to ONNX for lightweight serving")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/isl_mobilenet.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_image_model(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, checkpoint.get("image_size", 224), checkpoint.get("image_size", 224))
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    metadata = {
        "classes": checkpoint["classes"],
        "architecture": checkpoint.get("architecture", "MobileNetV3-Small"),
        "image_size": checkpoint.get("image_size", 224),
        "confidence_threshold": 0.60,
        "validation_accuracy": checkpoint.get("validation_accuracy"),
        "validation_macro_f1": checkpoint.get("validation_macro_f1"),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"ONNX model: {output}")
    print(f"Label metadata: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
