from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ml.datasets.image_dataset import make_image_folder
from ml.models.image_classifier import build_image_model

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def main():
    parser = argparse.ArgumentParser(description="Train SignBridge image-only ISL alphabet classifier")
    parser.add_argument("--data", required=True, help="ImageFolder root: one subfolder per class")
    parser.add_argument("--output", default="artifacts/isl_mobilenet.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet initialization")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.88, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_base = make_image_folder(args.data, train_transform)
    eval_base = make_image_folder(args.data, eval_transform)
    targets = np.asarray(train_base.targets)
    indices = np.arange(len(targets))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=args.seed, stratify=targets
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15 / 0.85,
        random_state=args.seed,
        stratify=targets[train_val_idx],
    )
    print(f"Classes: {train_base.classes}")
    print(f"Images: total={len(indices)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print("Evaluation warning: random image splits can leak near-duplicate/augmented images or signer backgrounds across splits. Prefer signer-independent source data for research claims.")

    train_loader = DataLoader(Subset(train_base, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(Subset(eval_base, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = build_image_model(len(train_base.classes), pretrained=not args.no_pretrained).to(device)
    except Exception as exc:
        if args.no_pretrained:
            raise
        print(f"Could not load pretrained ImageNet weights ({exc}); falling back to random initialization.")
        model = build_image_model(len(train_base.classes), pretrained=False).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        y_true, y_pred = [], []
        with torch.inference_mode():
            for images, labels in val_loader:
                predictions = model(images.to(device)).argmax(1).cpu().tolist()
                y_pred.extend(predictions)
                y_true.extend(labels.tolist())
        val_accuracy = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"epoch={epoch + 1:02d}/{args.epochs} loss={np.mean(losses):.4f} val_accuracy={val_accuracy:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint = {
                "state_dict": model.cpu().state_dict(),
                "classes": train_base.classes,
                "architecture": "MobileNetV3-Small",
                "image_size": IMAGE_SIZE,
                "mean": MEAN,
                "std": STD,
                "validation_accuracy": float(val_accuracy),
                "validation_macro_f1": float(val_f1),
                "seed": args.seed,
                "test_indices": test_idx.tolist(),
                "dataset_root": str(train_base.root),
            }
            torch.save(checkpoint, output)
            model.to(device)

    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"PyTorch checkpoint: {output}")
    print("Next: python -m ml.evaluation.evaluate_image --data <same-image-folder-root> --checkpoint " + str(output))
    print("Then export ONNX: python -m ml.training.export_onnx --checkpoint " + str(output) + " --output artifacts/isl_mobilenet.onnx")


if __name__ == "__main__":
    main()
