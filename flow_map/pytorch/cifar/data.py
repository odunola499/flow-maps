import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Iterator, Tuple


def get_transforms(image_size: int = 32):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_cifar10_loader(
    batch_size: int = 64,
    image_size: int = 32,
    num_workers: int = 4,
    train: bool = True,
    data_root: str = "./data",
) -> DataLoader:
    transform = get_transforms(image_size)
    dataset = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


class CIFAR10Dataset:
    def __init__(
        self,
        batch_size: int = 64,
        image_size: int = 32,
        num_workers: int = 4,
        data_root: str = "./data",
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.data_root = data_root

        self.train_loader = get_cifar10_loader(
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            train=True,
            data_root=data_root,
        )
        self.valid_loader = get_cifar10_loader(
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
            train=False,
            data_root=data_root,
        )

    def get_train_iterator(self) -> Iterator:
        return infinite_loader(self.train_loader)

    def get_valid_iterator(self) -> Iterator:
        return infinite_loader(self.valid_loader)


if __name__ == "__main__":
    dataset = CIFAR10Dataset(batch_size=32)
    train_iter = dataset.get_train_iterator()
    images, labels = next(train_iter)
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.2f}, {images.max():.2f}]")
