from pathlib import Path

from torchvision.datasets import ImageFolder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve_imagefolder_root(path: str | Path) -> Path:
    """Find a directory whose immediate subfolders are image classes.

    Supports a direct A-Z folder tree or one extra wrapper directory from an archive.
    Pass the specific Training folder when an archive contains train/test/validation.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    candidates = [root]
    candidates.extend(p for p in root.iterdir() if p.is_dir())
    valid = []
    for candidate in candidates:
        class_dirs = [p for p in candidate.iterdir() if p.is_dir()]
        image_classes = [
            p for p in class_dirs
            if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in p.rglob("*") if f.is_file())
        ]
        if len(image_classes) >= 2:
            valid.append((len(image_classes), candidate))
    if not valid:
        raise FileNotFoundError(
            f"Could not find class folders containing images under {root}. Expected root/A/*.jpg, root/B/*.jpg, etc."
        )
    return max(valid, key=lambda item: item[0])[1]


def make_image_folder(path: str | Path, transform=None) -> ImageFolder:
    root = resolve_imagefolder_root(path)
    dataset = ImageFolder(str(root), transform=transform)
    if len(dataset.classes) < 2:
        raise ValueError("At least two image classes are required.")
    return dataset
