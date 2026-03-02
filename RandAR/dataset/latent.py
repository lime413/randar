import torch
import os
import numpy as np

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
        latents = self.transform(latents) # 1 * aug_num * block_size

        # select one of the augmented crops
        aug_idx = torch.randint(0, latents.shape[1], (1,)).item()
        latents = latents[:, aug_idx, :]
        label = sample[1]
        
        return latents, label, index