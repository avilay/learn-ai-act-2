import os
from pathlib import Path

import torch as t
import torch.distributed as dist
import torch.nn.functional as F
import torcheval.metrics as teval
import torchvision as tv
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler  # noqa
from torcheval.metrics.toolkit import sync_and_compute_collection
from tqdm import tqdm

# Hyperparams
N_EPOCHS = 7
LOCAL_BATCH_SIZE = 64
DROPOUTS = [0.25, 0.5]
MOMENTUM = 0.9
LR = 0.001

# WORKSPACE = Path("/workspace")
WORKSPACE = Path.home()
DATAROOT = WORKSPACE / "mldata" / "mnist"
CHECKPOINT = WORKSPACE / "mlruns" / "dist_mnist_net.ckpt"


def dist_print(text):
    rank = dist.get_rank()
    pid = os.getpid()
    print(f"[{rank}]({pid}): {text}")


class Net(t.nn.Module):
    def __init__(self, dropouts):
        super().__init__()
        self.conv1 = t.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = t.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = t.nn.Dropout2d(dropouts[0])
        self.dropout2 = t.nn.Dropout2d(dropouts[1])
        self.fc1 = t.nn.Linear(9216, 128)
        self.fc2 = t.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def build_train_datasets(dataroot):
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5], [0.5])]
    )
    mnist = tv.datasets.MNIST(dataroot, train=True, transform=xform)
    train_size = int(len(mnist) * 0.9)
    trainset = Subset(mnist, range(train_size))
    valset = Subset(mnist, range(train_size, len(mnist)))
    return trainset, valset


def build_test_dataset(dataroot):
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5], [0.5])]
    )
    testset = tv.datasets.MNIST(dataroot, train=False, transform=xform)
    return testset


def build_dataloaders(dataroot) -> tuple[DataLoader, DataLoader]:
    trainset, valset = build_train_datasets(dataroot)
    # traindl = DataLoader(
    #     trainset,
    #     batch_size=LOCAL_BATCH_SIZE,
    #     sampler=DistributedSampler(trainset, shuffle=True),
    # )
    # valdl = DataLoader(
    #     valset, batch_size=1000, sampler=DistributedSampler(valset, shuffle=False)
    # )
    traindl = DataLoader(trainset, batch_size=LOCAL_BATCH_SIZE, shuffle=True)
    valdl = DataLoader(valset, batch_size=1000)
    return traindl, valdl


def train_loop(
    model: t.nn.Module,
    optim: t.optim.Optimizer,
    loss_fn: t.nn.Module,
    dl: DataLoader,
    metrics: dict[str, teval.Metric],
) -> dict[str, float]:
    model.train()
    with t.enable_grad():
        for inputs, targets in tqdm(dl):
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optim.step()

            metrics["acc"].update(outputs, targets)
            metrics["loss"].update(loss.detach())

    metric_values = sync_and_compute_collection(metrics)
    metrics["acc"].reset()
    metrics["loss"].reset()

    return metric_values


def eval_loop(
    model: t.nn.Module,
    loss_fn: t.nn.Module,
    dl: DataLoader,
    metrics: dict[str, teval.Metric],
) -> dict[str, float]:
    model.eval()
    with t.no_grad():
        for inputs, targets in dl:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            metrics["acc"].update(outputs, targets)
            metrics["loss"].update(loss.detach())

    metric_values = sync_and_compute_collection(metrics)
    metrics["acc"].reset()
    metrics["loss"].reset()
    return metric_values


def log_metrics(
    epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]
):
    print(f"Epoch {epoch}:")
    print(
        "Train: loss={:.3f} acc={:.3f}".format(
            train_metrics["loss"], train_metrics["acc"]
        )
    )
    print("Val: loss={:.3f} acc={:.3f}".format(val_metrics["loss"], val_metrics["acc"]))


def main():
    # print("Starting process group")
    # dist.init_process_group()

    train_acc_metric = teval.MulticlassAccuracy()
    train_loss_metric = teval.Mean()

    val_acc_metric = teval.MulticlassAccuracy()
    val_loss_metric = teval.Mean()

    model = Net(DROPOUTS)
    # model = DDP(model)
    optim = t.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loss_fn = t.nn.NLLLoss()

    train_dl, val_dl = build_dataloaders(DATAROOT)

    for epoch in range(N_EPOCHS):
        train_metrics = train_loop(
            model,
            optim,
            loss_fn,
            train_dl,
            {"acc": train_acc_metric, "loss": train_loss_metric},
        )

        val_metrics = eval_loop(
            model, loss_fn, val_dl, {"acc": val_acc_metric, "loss": val_loss_metric}
        )

        log_metrics(epoch, train_metrics, val_metrics)


if __name__ == "__main__":
    main()
