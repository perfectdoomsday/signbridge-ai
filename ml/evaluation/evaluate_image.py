from __future__ import annotations

import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ml.datasets.image_dataset import make_image_folder
from ml.models.image_classifier import build_image_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate SignBridge image classifier on its held-out test split")
    parser.add_argument("--data", required=True, help="Same ImageFolder root passed to training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(checkpoint.get("image_size", 224)),
        transforms.ToTensor(), transforms.Normalize(checkpoint.get("mean", (0.485, 0.456, 0.406)), checkpoint.get("std", (0.229, 0.224, 0.225))),
    ])
    dataset = make_image_folder(args.data, transform)
    if dataset.classes != checkpoint["classes"]:
        raise ValueError(f"Class order mismatch. Checkpoint={checkpoint['classes']} dataset={dataset.classes}")
    test_indices = checkpoint.get("test_indices")
    if not test_indices:
        raise ValueError("Checkpoint has no held-out test indices; retrain with the current training script.")
    loader = DataLoader(Subset(dataset, test_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = build_image_model(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            y_pred.extend(model(images).argmax(1).tolist())
            y_true.extend(labels.tolist())
    print("Held-out test split from the same image collection; not signer-independent unless the source data split guarantees that.")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, target_names=dataset.classes, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
