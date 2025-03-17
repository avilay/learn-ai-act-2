from datetime import datetime, timezone
from pathlib import Path

import click
import torch as t
import torcheval.metrics as teval
from haikunator import Haikunator
from torch.utils.data import DataLoader
from torchsnapshot import Snapshot, StateDict
from tqdm import tqdm

from mnist.dataset import build_train_datasets
from mnist.model import Net

# Hyperparams
N_EPOCHS = 14
# BATCH_SIZE = 64
BATCH_SIZE = 256
DROPOUTS = (0.25, 0.5)
MOMENTUM = 0.9
LR = 0.001

WORKSPACE = Path.home()
DATAROOT = WORKSPACE / "mldata" / "pytorch"
CHECKPOINT = WORKSPACE / "mlruns" / "mnist" / "{}_{}.ckpt"


def build_dataloaders(dataroot: Path) -> tuple[DataLoader, DataLoader]:
    trainset, valset = build_train_datasets(dataroot)
    traindl = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valdl = DataLoader(valset, batch_size=1000)
    return traindl, valdl


def train_epoch(
    model: t.nn.Module,
    optim: t.optim.Optimizer,
    loss_fn: t.nn.Module,
    dl: DataLoader,
    metrics: dict[str, teval.Metric],
) -> dict[str, float]:
    acc_metric, loss_metric = metrics["acc"], metrics["loss"]
    model.train()
    with t.enable_grad():
        for inputs, targets in tqdm(dl):
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optim.step()

            acc_metric.update(outputs, targets)
            loss_metric.update(loss.detach())

    acc = acc_metric.compute()
    loss = loss_metric.compute()
    acc_metric.reset()
    loss_metric.reset()
    return {"acc": acc, "loss": loss}


def eval_epoch(
    model: t.nn.Module,
    loss_fn: t.nn.Module,
    dl: DataLoader,
    metrics: dict[str, teval.Metric],
) -> dict[str, float]:
    acc_metric, loss_metric = metrics["acc"], metrics["loss"]
    model.eval()
    with t.no_grad():
        for inputs, targets in dl:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            acc_metric.update(outputs, targets)
            loss_metric.update(loss.detach())

    acc = acc_metric.compute()
    loss = loss_metric.compute()
    acc_metric.reset()
    loss_metric.reset()
    return {"acc": acc, "loss": loss}


def log_metrics(
    epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]
) -> None:
    print(f"Epoch {epoch}:")

    train_loss, train_acc = train_metrics["loss"], train_metrics["acc"]
    print(f"Train: loss={train_loss:.3f} acc={train_acc:.3f}")

    val_loss, val_acc = val_metrics["loss"], val_metrics["acc"]
    print(f"Val: loss={val_loss:.3f} acc={val_acc:.3f}")


@click.command()
@click.option("--ckpt", default="", help="Checkpoint path.")
@click.option("--name", default="", help="Name of this training run.")
def train(ckpt="", name="") -> None:
    name = name or Haikunator().haikunate()
    print(f"Starting training run: {name}")
    train_acc_metric = teval.MulticlassAccuracy()
    train_loss_metric = teval.Mean()

    val_acc_metric = teval.MulticlassAccuracy()
    val_loss_metric = teval.Mean()

    model = Net(DROPOUTS)
    optim = t.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loss_fn = t.nn.NLLLoss()

    if ckpt and Path(ckpt).exists():
        print(f"Restoring checkpoint from {ckpt}.")
        snapshot = Snapshot(path=ckpt)
        app_state = {"model": model, "optim": optim, "epoch": StateDict(epoch=-1)}
        snapshot.restore(app_state=app_state)
        start_epoch = app_state["epoch"]["epoch"] + 1
    else:
        start_epoch = 0

    train_dl, val_dl = build_dataloaders(DATAROOT)

    try:
        for epoch in range(start_epoch, N_EPOCHS):
            train_metrics = train_epoch(
                model,
                optim,
                loss_fn,
                train_dl,
                {"acc": train_acc_metric, "loss": train_loss_metric},
            )
            val_metrics = eval_epoch(
                model, loss_fn, val_dl, {"acc": val_acc_metric, "loss": val_loss_metric}
            )
            log_metrics(epoch, train_metrics, val_metrics)
            ts = int(datetime.now(tz=timezone.utc).timestamp())
            Snapshot.take(
                str(CHECKPOINT).format(name, ts),
                app_state={
                    "model": model,
                    "optim": optim,
                    "epoch": StateDict(epoch=epoch),
                },
            )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    train()
