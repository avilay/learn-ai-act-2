import click
from haikunator import Haikunator
from torchtnt.utils import (
    seed as set_seed,
    init_from_env,
    TLRScheduler,
    copy_data_to_device,
)

# from torchtnt.utils.loggers import CSVLogger
from pathlib import Path
from mnist.model import Net
import torch as t
from torcheval.metrics import MulticlassAccuracy
from mnist.dataset import build_train_datasets
from torch.utils.data import DataLoader
from torchtnt.framework.unit import TrainUnit
from torchtnt.framework.train import train as tnttrain
from torchtnt.framework.state import State

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


def build_dataloaders(dataroot: Path) -> tuple[DataLoader, DataLoader]:
    trainset, valset = build_train_datasets(dataroot)
    traindl = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valdl = DataLoader(valset, batch_size=1000)
    return traindl, valdl


class MnistTrainer(TrainUnit[Batch]):
    def __init__(
        self,
        model: t.nn.Module,
        optim: t.optim.Optimizer,
        loss_fn: t.nn.Module,
        lr_scheduler: TLRScheduler,
        device: t.device,
        train_acc: MulticlassAccuracy,
        # logger: CSVLogger,
    ):
        super().__init__()
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.train_acc = train_acc
        # self.logger = logger

    def train_step(self, state: State, data: Batch) -> None:
        inputs, targets = copy_data_to_device(data, self.device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optim.step()
        # self.model.train()
        # with t.enable_grad():
        #     self.optim.zero_grad()
        self.train_acc.update(outputs, targets)

    def on_train_epoch_end(self, state: State) -> None:
        self.lr_scheduler.step()

        acc = self.train_acc.compute()
        self.train_acc.reset()
        # self.logger.log(
        #     name="acc_epoch", data=acc, step=self.train_progress.num_epochs_completed
        # )
        epoch = self.train_progress.num_epochs_completed
        print(f"Train Epoch {epoch}: acc = {acc:.3f}")


@click.command()
@click.option("--name", default="", help=".")
@click.option("--seed", default=1, help=".")
def main(name: str, seed: int):
    name = name or Haikunator().haikunate()
    print(f"Starting training run {name}")

    set_seed(seed)

    device = init_from_env()

    # logpath = RUNROOT / f"{name}.csv"
    # ml_logger = CSVLogger(path=str(logpath))

    train_acc = MulticlassAccuracy()

    model = Net(DROPOUTS).to(device)
    optim = t.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loss_fn = t.nn.NLLLoss()
    lr_scheduler = t.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)

    train_dl, _ = build_dataloaders(DATAROOT)
    model.eval()
    t.set_grad_enabled(False)
    trainer = MnistTrainer(
        model,
        optim,
        loss_fn,
        lr_scheduler,
        device,
        train_acc,  # ml_logger
    )
    tnttrain(trainer, train_dl, max_epochs=N_EPOCHS)


if __name__ == "__main__":
    main()
