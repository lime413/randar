import argparse
import shutil
from pathlib import Path
import numpy as np


def split_imagenet256_latents(
    source_root: Path,
    target_root: Path,
    train_per_class: int = 210,
    val_per_class: int = 60,
    test_per_class: int = 30,
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

    class_dirs = sorted([d for d in source_root.iterdir() if d.is_dir()], key=lambda d: int(d.name))
    if len(class_dirs) == 0:
        raise ValueError(f"No class subdirectories found in {source_root}")

    for class_dir in class_dirs:
        files = sorted([p for p in class_dir.iterdir() if p.suffix == ".npy"])
        if len(files) < train_per_class + val_per_class + test_per_class:
            raise ValueError(
                f"Class {class_dir.name} contains only {len(files)} .npy files, "
                f"but {train_per_class + val_per_class + test_per_class} are required."
            )

        chosen = rng.permutation(len(files))
        train_idxs = chosen[:train_per_class]
        val_idxs = chosen[train_per_class : train_per_class + val_per_class]
        test_idxs = chosen[train_per_class + val_per_class : train_per_class + val_per_class + test_per_class]

        for subset_name, indices in [
            ("train", train_idxs),
            ("val", val_idxs),
            ("test", test_idxs),
        ]:
            subset_root = target_root / subset_name / class_dir.name
            subset_root.mkdir(parents=True, exist_ok=True)
            for idx in indices:
                src_path = files[int(idx)]
                dst_path = subset_root / src_path.name
                shutil.copy2(src_path, dst_path)

    print(f"Saved splits to: {target_root}")
    print(f"train: {train_per_class} files/class, val: {val_per_class} files/class, test: {test_per_class} files/class")
    print(f"Total train samples: {len(class_dirs) * train_per_class}")
    print(f"Total val samples: {len(class_dirs) * val_per_class}")
    print(f"Total test samples: {len(class_dirs) * test_per_class}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split ImageNet256 latent .npy files into train/val/test folders."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="data/imagenet256/latents",
        help="Root directory containing class subfolders with .npy files.",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default="data/imagenet256-splits",
        help="Target directory to create train/val/test split folders.",
    )
    parser.add_argument("--train-per-class", type=int, default=210)
    parser.add_argument("--val-per-class", type=int, default=60)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    split_imagenet256_latents(
        source_root=Path(args.source_root),
        target_root=Path(args.target_root),
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )
