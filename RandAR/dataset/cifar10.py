from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image

class CIFAR10WithIndex(Dataset):
    """
    Wraps torchvision.datasets.CIFAR10 to return (image, label, index)
    """
    def __init__(self, root, train=True, transform=None, download=False):
        self.ds = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
        self.nb_classes = 10

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        return img, label, index


class CIFAR10CSeverityDataset(Dataset):
    """
    Dataset for one CIFAR-10-C severity folder.
    A random subset is taken from each corruption so total size is near target_total_size
    """

    def __init__(
        self,
        severity_dir,
        labels_path,
        transform=None,
        target_total_size=10000,
        seed=42,
    ):
        self.severity_dir = Path(severity_dir)
        self.labels_path = Path(labels_path)
        self.severity = int(severity_dir.split('/')[-1].split('_')[-1])
        self.transform = transform
        self.target_total_size = int(target_total_size)
        self.seed = int(seed)
        self.nb_classes = 10

        if self.severity not in [1, 2, 3, 4, 5]:
            raise ValueError(f"severity must be in [1, 5], got {self.severity}")

        if not self.severity_dir.exists():
            raise FileNotFoundError(f"Severity folder not found: {self.severity_dir}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        full_labels = np.load(self.labels_path)
        if full_labels.ndim != 1:
            raise ValueError(f"labels.npy should be 1D, got shape {full_labels.shape}")
        if len(full_labels) != 50000:
            raise ValueError(f"Expected labels.npy of length 50000, got {len(full_labels)}")

        start = (self.severity - 1) * 10000
        end = self.severity * 10000
        self.labels = full_labels[start:end]

        self.corruption_files = sorted(
            [p for p in self.severity_dir.glob("*.npy") if p.name != "labels.npy"]
        )
        if len(self.corruption_files) == 0:
            raise ValueError(f"No corruption .npy files found in {self.severity_dir}")

        self.arrays = [np.load(path, mmap_mode="r") for path in self.corruption_files]

        self.samples_per_corruption_full = 10000

        for path, arr in zip(self.corruption_files, self.arrays):
            if arr.shape[0] != self.samples_per_corruption_full:
                raise ValueError(
                    f"Mismatch in {path.name}: expected first dimension "
                    f"{self.samples_per_corruption_full}, got {arr.shape[0]}"
                )

        self.num_corruptions = len(self.arrays)

        max_possible = self.num_corruptions * self.samples_per_corruption_full
        self.target_total_size = min(max(1, self.target_total_size), max_possible)

        base = self.target_total_size // self.num_corruptions
        remainder = self.target_total_size % self.num_corruptions

        rng = np.random.default_rng(self.seed)

        self.index_map = []
        for corruption_idx in range(self.num_corruptions):
            take_n = base + (1 if corruption_idx < remainder else 0)
            take_n = min(take_n, self.samples_per_corruption_full)

            chosen_indices = rng.choice(
                self.samples_per_corruption_full,
                size=take_n,
                replace=False,
            )
            chosen_indices = np.sort(chosen_indices)

            for sample_idx in chosen_indices:
                self.index_map.append((corruption_idx, int(sample_idx)))

        self.total_len = len(self.index_map)

        print(f"[CIFAR10CSeveritySubsetDataset] severity_dir={self.severity_dir}")
        print(f"[CIFAR10CSeveritySubsetDataset] severity={self.severity}")
        print(f"[CIFAR10CSeveritySubsetDataset] num_corruptions={self.num_corruptions}")
        print(f"[CIFAR10CSeveritySubsetDataset] full samples/corruption={self.samples_per_corruption_full}")
        print(f"[CIFAR10CSeveritySubsetDataset] target_total_size={self.target_total_size}")
        print(f"[CIFAR10CSeveritySubsetDataset] actual_total_size={self.total_len}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if index < 0 or index >= self.total_len:
            raise IndexError(f"Index {index} is out of range for dataset of size {self.total_len}")

        corruption_idx, sample_idx = self.index_map[index]

        img = self.arrays[corruption_idx][sample_idx]
        label = int(self.labels[sample_idx])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, index