import os
from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms


class INatLatentDataset(data.Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        self.categories = sorted([int(i) for i in list(os.listdir(root_dir))])
        self.samples = []

        for tgt_class in self.categories:
            tgt_dir = os.path.join(root_dir, str(tgt_class))
            for root, _, fnames in sorted(os.walk(tgt_dir, followlinks=True)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    item = (path, tgt_class)
                    self.samples.append(item)
        self.num_examples = len(self.samples)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(root_dir, self.num))
        self.transform = transform
  
    def __len__(self):
        return self.num_examples
  
    def __getitem__(self, index):
        index = self.indices[index]
        sample = self.samples[index]
        latents = np.load(sample[0])

        if self.transform is not None:
            latents = self.transform(latents)
        else:
            latents = torch.from_numpy(latents).long()

        if latents.dim() == 3:
            if latents.shape[1] > 1:
                aug_idx = torch.randint(0, latents.shape[1], (1,)).item()
                latents = latents[:, aug_idx, :]
            else:
                latents = latents[:, 0, :]

        if latents.dim() == 1:
            latents = latents.unsqueeze(0)

        label = sample[1]
        return latents, label, index


class ImageNet256LatentDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")

        self.samples = []
        self.categories = set()

        for npy_path in sorted(self.root_dir.rglob("*.npy")):
            if not npy_path.is_file():
                continue

            rel_path = npy_path.relative_to(self.root_dir)
            if len(rel_path.parts) < 2:
                raise ValueError(
                    f"Expected class subdirectory under {self.root_dir}, "
                    f"but found .npy file at {npy_path}."
                )

            label_str = rel_path.parts[0]
            try:
                label = int(label_str)
            except ValueError as exc:
                raise ValueError(
                    f"Class folder names must be integer labels, got '{label_str}'"
                ) from exc

            self.samples.append((str(npy_path), label))
            self.categories.add(label)

        if len(self.samples) == 0:
            raise ValueError(f"No .npy files found under {self.root_dir}")

        self.categories = sorted(self.categories)
        self.num_examples = len(self.samples)
        self.indices = np.arange(self.num_examples)
        self.num = self.num_examples

        print(
            f"Loaded ImageNet latent dataset from {self.root_dir}. "
            f"Using {self.num_examples} samples from {len(self.categories)} classes."
        )

        self.transform = transform
        self.nb_classes = len(self.categories)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        index = self.indices[index]
        path, label = self.samples[index]
        latents = np.load(path)

        if self.transform is not None:
            latents = self.transform(latents)
        else:
            latents = torch.from_numpy(latents).long()

        if latents.dim() == 3:
            if latents.shape[1] > 1:
                aug_idx = torch.randint(0, latents.shape[1], (1,)).item()
                latents = latents[:, aug_idx, :]
            else:
                latents = latents[:, 0, :]

        if latents.dim() == 1:
            latents = latents.unsqueeze(0)

        return latents, label, index
