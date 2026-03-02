import os

def build_dataset(is_train, args, transform):
    if args.dataset == "imagenet":
        from .imagenet import ImageTarDataset
        root = os.path.join(args.data_path, "train.tar" if is_train else "val.tar")
        dataset = ImageTarDataset(root, return_labels=True, transform=transform)
        dataset.nb_classes = 1000

    elif args.dataset == "cifar10":
        from .cifar10 import CIFAR10WithIndex

        dataset = CIFAR10WithIndex(
            root=args.data_path,
            train=is_train,
            transform=transform,
            download=False,
        )
        dataset.nb_classes = 10

    elif args.dataset == "latent":
        from .imagenet import INatLatentDataset
        dataset = INatLatentDataset(root_dir=args.data_path, transform=transform)

    else:
        raise NotImplementedError

    return dataset
