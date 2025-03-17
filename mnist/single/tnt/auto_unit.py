from pathlib import Path
from typing import Any, Tuple

import click
import pretty_traceback
import torch as t
from haikunator import Haikunator
from mnist.dataset import build_train_datasets
from mnist.model import Net
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torcheval.metrics import Mean, MulticlassAccuracy
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.fit import fit
from torchtnt.framework.state import State
from torchtnt.utils import copy_data_to_device
from torchtnt.utils import seed as set_seed
from torchtnt.utils.loggers import InMemoryLogger, MetricLogger

pretty_traceback.install()

Batch = tuple[t.Tensor, t.Tensor]

# Hyperparams
N_EPOCHS = 5
BATCH_SIZE = 64
DROPOUTS = (0.25, 0.5)
MOMENTUM = 0.9
LR = 0.001

WORKSPACE = Path.home()
RUNROOT = WORKSPACE / "mlruns" / "learn-torchtnt"
DATAROOT = WORKSPACE / "mldata" / "pytorch"


class MnistTrainer(AutoUnit):
    def __init__(self, model: t.nn.Module, logger: MetricLogger):
        super().__init__(module=model)
        self.loss_fn = t.nn.NLLLoss()
        self.train_acc, self.val_acc = MulticlassAccuracy(), MulticlassAccuracy()
        self.train_losses, self.val_losses = Mean(), Mean()
        self.logger = logger

    def compute_loss(self, state: State, data: Batch) -> tuple[t.Tensor, Any]:
        inputs, targets = copy_data_to_device(data, self.device)
        outputs = self.module(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss, outputs

    def configure_optimizers_and_lr_scheduler(
        self, module: t.nn.Module
    ) -> Tuple[t.optim.Optimizer, LRScheduler | None]:
        optim = t.optim.SGD(self.module.parameters(), lr=LR, momentum=MOMENTUM)
        lr_scheduler = t.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
        return optim, lr_scheduler

    def on_train_step_end(
        self, state: State, data: Batch, step: int, loss: t.Tensor, outputs: t.Tensor
    ) -> None:
        _, targets = data
        self.train_acc.update(outputs, targets)
        self.train_losses.update(loss.detach())

    def on_train_epoch_end(self, state: State) -> None:
        acc = self.train_acc.compute()
        loss = self.train_losses.compute()
        self.train_acc.reset()
        self.train_losses.reset()

        self.logger.log_dict(
            {"train_epoch_accuracy": acc, "train_epoch_loss": loss},
            step=self.train_progress.num_epochs_completed,
        )

        self.logger.flush()

    def on_eval_step_end(
        self, state: State, data: Batch, step: int, loss: t.Tensor, outputs: t.Tensor
    ) -> None:
        _, targets = data
        self.val_acc.update(outputs, targets)
        self.val_losses.update(loss.detach())

    def on_eval_epoch_end(self, state: State) -> None:
        acc = self.val_acc.compute()
        loss = self.val_losses.compute()
        self.val_acc.reset()
        self.val_losses.reset()

        self.logger.log_dict(
            {"val_epoch_accuracy": acc, "val_epoch_loss": loss},
            step=self.eval_progress.num_epochs_completed,
        )


def build_dataloaders(dataroot: Path) -> tuple[DataLoader, DataLoader]:
    trainset, valset = build_train_datasets(dataroot)
    train_dl = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(valset, batch_size=1000)
    return train_dl, val_dl


@click.command()
@click.option("--name", default="", help="Name of the run.")
@click.option("--seed", default=1, help="Random seed to use")
def main(name: str, seed: int):
    name = name or Haikunator().haikunate()
    print(f"Starting training run {name}")

    set_seed(seed)

    model = Net(DROPOUTS)  # Will AutUnit load the model onto the device?

    train_dl, val_dl = build_dataloaders(DATAROOT)

    # logpath = RUNROOT / f"{name}.csv"
    # ml_logger = CSVLogger(path=str(logpath), steps_before_flushing=1)
    ml_logger = InMemoryLogger()

    trainer = MnistTrainer(model=model, logger=ml_logger)
    fit(trainer, train_dl, val_dl, max_epochs=N_EPOCHS)
    ml_logger.close()


if __name__ == "__main__":
    main()
