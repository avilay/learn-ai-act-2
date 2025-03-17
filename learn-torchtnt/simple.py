import torch as t
from torch.utils.data import DataLoader, Dataset
from torchtnt.framework.auto_unit import AutoUnit
from torchtnt.framework.fit import fit
from torchtnt.framework.callbacks import TorchSnapshotSaver


class Simple(AutoUnit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctr = 0

    def configure_optimizers_and_lr_scheduler(self, model):
        optim = t.optim.SGD(model.parameters(), lr=0.1)
        return optim, None

    def compute_loss(self, state, batch):
        self.ctr += 1
        print(f"{self.ctr}. compute_loss")
        inputs, targets = batch
        outputs = self.module(inputs)
        outputs = outputs.flatten()
        loss = t.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
        return loss, outputs

    def on_train_step_end(self, state, data, step, loss, outputs):
        self.ctr += 1
        print(f"{self.ctr}. on_train_step_end")

    def on_train_epoch_end(self, state):
        self.ctr += 1
        print(f"{self.ctr}. on_train_epoch_end")

    def on_eval_step_end(self, state, data, step, loss, outputs):
        self.ctr += 1
        print(f"{self.ctr}. on_eval_step_end")

    def on_train_end(self, state):
        self.ctr += 1
        print(f"{self.ctr}. on_train_end")


class MyDataset(Dataset):
    def __init__(self):
        self._x = t.rand((10, 3))
        self._y = t.randint(0, 2, (10,)).to(t.float32)

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]

    def __len__(self):
        return 10


def main():
    print("Starting to fit")
    train_dl = DataLoader(MyDataset(), batch_size=2)
    val_dl = DataLoader(MyDataset(), batch_size=10)
    model = t.nn.Linear(3, 1)
    trainer = Simple(module=model)
    checkpointer = TorchSnapshotSaver(
        dirpath="/Users/avilay/mlruns/learn-torchtnt/checkpoints/"
    )
    fit(trainer, train_dl, val_dl, max_epochs=1, callbacks=[checkpointer])
    print("Done")


if __name__ == "__main__":
    main()
