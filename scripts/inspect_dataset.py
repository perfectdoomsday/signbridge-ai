from collections import Counter
from pathlib import Path
import argparse
import sys

from ml.datasets.image_dataset import IMAGE_EXTENSIONS, resolve_imagefolder_root

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Inspect the SignBridge image dataset folder layout")
    parser.add_argument("path", nargs="?", default="data/raw/ISL")
    args = parser.parse_args()
    root = resolve_imagefolder_root(args.path)
    counts = Counter()
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        counts[class_dir.name] = sum(
            1 for file in class_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )
    print(f"Detected ImageFolder root: {root}")
    print(f"Classes: {len(counts)} | Images: {sum(counts.values())}")
    for label, count in sorted(counts.items()):
        print(f"{label:>12}: {count}")


if __name__ == "__main__":
    main()
