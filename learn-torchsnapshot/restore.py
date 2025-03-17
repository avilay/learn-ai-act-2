import torch as t
from torchsnapshot import Snapshot, RNGState
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, RandomSampler


def test_rng() -> None:
    app_state = {"rng_state": RNGState()}
    snapshot = Snapshot(path="./states.tss")
    # No need to set the rng state with t.random.set_rng_state(), RNGState does all the magic.
    snapshot.restore(app_state=app_state)
    print(t.rand(5, 1))


def test_model() -> None:
    model = t.nn.Linear(3, 1)
    app_state = {"model": model}
    print("Before restore -")
    [print(param) for param in model.parameters()]
    snapshot = Snapshot(path="./states.tss")
    snapshot.restore(app_state=app_state)
    print("\nAfter restore -")
    [print(param) for param in model.parameters()]


def test_model_rng() -> None:
    """
    Here I am not restoring the model, but still the random initialization
    will be the same as in save.py.
    """
    app_state = {"rng_state": RNGState()}
    snapshot = Snapshot(path="./states.tss")
    snapshot.restore(app_state=app_state)
    model = t.nn.Linear(3, 1)
    [print(param) for param in model.parameters()]


def test_data() -> None:
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
    ckpt_path = Path.home() / "temp" / "state.tss"
    app_state = {"rng_state": RNGState(), "sampler": RandomSampler(ds)}
    snapshot = Snapshot(path=str(ckpt_path))
    snapshot.restore(app_state=app_state)
    dl = DataLoader(dataset=ds, batch_size=2, sampler=app_state["sampler"])
    for step, (x, y) in enumerate(dl):
        print(f"[step]: {x}, {y}")


def main():
    # test_rng()
    test_model()
    # test_model_rng()
    # test_data()


if __name__ == "__main__":
    main()
