from pathlib import Path
from torch.utils.data import Dataset, Subset
import torchvision as tv


def build_train_datasets(dataroot: Path) -> tuple[Dataset, Dataset]:
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5], [0.5])]
    )
    mnist = tv.datasets.MNIST(str(dataroot), train=True, download=True, transform=xform)
    train_size = int(len(mnist) * 0.9)
    trainset = Subset(mnist, range(train_size))
    valset = Subset(mnist, range(train_size, len(mnist)))
    return trainset, valset


def build_test_dataset(dataroot: Path) -> Dataset:
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5], [0.5])]
    )
    testset = tv.datasets.MNIST(dataroot, train=False, download=True, transform=xform)
    return testset
