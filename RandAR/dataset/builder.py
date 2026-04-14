import os
from pathlib import Path

def build_dataset(is_train, args, transform, split = None):
    if args.dataset == "imagenet":
        from .imagenet import ImageTarDataset
        root = os.path.join(args.data_path, "train.tar" if is_train else "val.tar")
        dataset = ImageTarDataset(root, return_labels=True, transform=transform)
        dataset.nb_classes = 1000

    elif  args.dataset == "cifar10_split":
        from .cifar10 import CIFAR10_split

        dataset = CIFAR10_split(
            root=args.data_path,
            split= split,
            transform=transform,
        )
        dataset.nb_classes = 10


    elif args.dataset == "cifar10":
        from .cifar10 import CIFAR10WithIndex

        dataset = CIFAR10WithIndex(
            root=args.data_path,
            train=is_train,
            transform=transform,
            download=False,
        )
        dataset.nb_classes = 10

    elif args.dataset == "cifar10c":
        from .cifar10 import CIFAR10CSeverityDataset

        p = Path(args.data_path)
        parent_dir = p.parent

        dataset = CIFAR10CSeverityDataset(
            severity_dir=args.data_path,
            labels_path=os.path.join(parent_dir, "labels.npy"),
            transform=transform,
        )
        dataset.nb_classes = 10

    elif args.dataset == "cifar10_latent":
        from .latent import INatLatentDataset
        if is_train:
            dataset = INatLatentDataset(root_dir=args.data_path, transform=None)
        else:
            dataset = INatLatentDataset(root_dir=args.val_path, transform=None)
        dataset.nb_classes = 10

    elif args.dataset in ["imagenet256_latent", "latent", "imagenet256-splits", "imagenet256_splits"]:
        from .latent import ImageNet256LatentDataset
        dataset = ImageNet256LatentDataset(root_dir=args.data_path, transform=None)
        dataset.nb_classes = 1000

    else:
        raise NotImplementedError

    return dataset
