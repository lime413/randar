import argparse
import shutil
from pathlib import Path
import numpy as np


def split_cifar10(
    source_root: Path,
    target_root: Path,
    train_per_class: int = 4000,
    val_per_class: int = 1000,
    seed: int = 42,
):
    source_root = Path(source_root)
    target_root = Path(target_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    train_root = target_root / "train"
    val_root = target_root / "val"
    test_root = target_root / "test"

    for path in [train_root, val_root, test_root]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    class_dirs = sorted([d for d in source_root.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"No class subdirectories found in {source_root}")

    print(f"Found {len(class_dirs)} classes")

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    for class_dir in class_dirs:
        files = sorted([p for p in class_dir.iterdir() 
                       if p.suffix.lower() in image_extensions])
        
        total_files = len(files)
        required_files = train_per_class + val_per_class
        
        if total_files < required_files:
            raise ValueError(
                f"Class {class_dir.name} has only {total_files} images, "
                f"but {required_files} are required."
            )

        chosen = rng.permutation(total_files)
        train_idxs = chosen[:train_per_class]
        val_idxs = chosen[train_per_class:train_per_class + val_per_class]

        for subset_name, indices in [
            ("train", train_idxs),
            ("val", val_idxs),
        ]:
            subset_root = target_root / subset_name / class_dir.name
            subset_root.mkdir(parents=True, exist_ok=True)
            
            for idx in indices:
                src_path = files[int(idx)]
                dst_path = subset_root / src_path.name
                shutil.copy2(src_path, dst_path)

    test_source = source_root.parent / "test"
    if test_source.exists():
        print("Processing test folder")
        test_class_dirs = sorted([d for d in test_source.iterdir() if d.is_dir()])
        for class_dir in test_class_dirs:
            files = sorted([p for p in class_dir.iterdir() 
                           if p.suffix.lower() in image_extensions])
            dst_class_root = test_root / class_dir.name
            dst_class_root.mkdir(parents=True, exist_ok=True)
            for file_path in files:
                dst_path = dst_class_root / file_path.name
                shutil.copy2(file_path, dst_path)
        print(f"Test samples: {sum(1 for _ in test_root.rglob('*.*') if _.suffix.lower() in image_extensions)}")

    print(f"Saved splits to: {target_root}")
    print(f"Train: {train_per_class} per class, {len(class_dirs) * train_per_class} total")
    print(f"Val: {val_per_class} per class, {len(class_dirs) * val_per_class} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split CIFAR-10 into train/val/test folders."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="data/cifar10/train",
        help="Source train folder with class subfolders.",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default="data/cifar10_split",
        help="Target directory for train/val/test splits.",
    )
    parser.add_argument("--train-per-class", type=int, default=4000)
    parser.add_argument("--val-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_cifar10(
        source_root=Path(args.source_root),
        target_root=Path(args.target_root),
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        seed=args.seed,
    )