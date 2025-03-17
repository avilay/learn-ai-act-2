import click
import torch as t
from torchsnapshot import Snapshot, RNGState, StateDict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def test_rng(seed: int | None) -> None:
    if seed is not None:
        t.random.manual_seed(seed)
    print(t.rand(5, 1))
    # No need to get t.random.get_rng_state() before hand, RNGState does all the magic.
    app_state = {"rng_state": RNGState()}
    Snapshot.take("./states.tss", app_state=app_state)
    print(t.rand((5, 1)))


def test_model() -> None:
    """
    I am only taking the snapshot of the model without any RNG magic.
    As expected the model will be restored with exactly these same weights.
    """
    model = t.nn.Linear(3, 1)
    [print(param) for param in model.parameters()]
    app_state = {"model": model}
    Snapshot.take("./states.tss", app_state=app_state)


def test_model_rng() -> None:
    """
    Here I will not checkpoint the model, but because of the RNGState
    magic, a new randomly initialized layer will be the same as after
    restore.
    """
    app_state = {"rng_state": RNGState()}
    Snapshot.take("./states.tss", app_state=app_state)
    model = t.nn.Linear(3, 1)
    [print(param) for param in model.parameters()]


def test_data() -> None:
    ckpt_path = Path.home() / "temp" / "state.tss"

    class MyDataset(Dataset):
        def __init__(self):
            self._x = []
            for i in range(10):
                self._x.append(t.full((3, 1), i).to(t.float32).flatten())
            self._y = t.randint(0, 2, (10,))

        def __getitem__(self, idx):
            return self._x[idx], self._y[idx]

        def __len__(self):
            return 10

    ds = MyDataset()
    dl = DataLoader(dataset=ds, batch_size=2, shuffle=True)
    for step, (x, y) in enumerate(dl):
        print(f"[step]: {x}, {y}")
        if step == 2:
            print("Taking snapshot of the rng state and sampler")
            app_state = {
                "rng_state": RNGState(),
                "sampler": StateDict(sampler=dl.sampler),
            }
            Snapshot.take(path=str(ckpt_path), app_state=app_state)


@click.command()
@click.option("--seed", default=None, type=int, help="Set the random generator seed")
def main(seed: int | None):
    # test_rng(seed)
    test_model()
    # test_model_rng()
    # test_data()


if __name__ == "__main__":
    main()
