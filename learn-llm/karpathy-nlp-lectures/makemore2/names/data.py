from torch.utils.data import Dataset
from pathlib import Path
import torch as t

EOT_TOKEN = "."


class NamesDataset(Dataset):
    def __init__(self, filepath: Path, context_len: int) -> None:
        # Vocab is pretty simple
        self._chars: list[str] = [EOT_TOKEN] + [chr(ord("a") + n) for n in range(26)]
        self._idxs: dict[str, int] = {char: idx for idx, char in enumerate(self._chars)}

        allchars: list[str] = []
        with open(filepath, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    line += EOT_TOKEN
                    allchars += line

        X: list[list[int]] = []
        y: list[int] = []
        context = [self._idxs[EOT_TOKEN]] * context_len
        for char in allchars:
            X.append(context)
            idx = self._idxs[char]
            y.append(idx)
            if char == EOT_TOKEN:
                context = [idx] * context_len
            else:
                context = context[1:] + [idx]
        self._X: t.Tensor = t.tensor(X)  # \in (-1 x context_len)
        self._y: t.Tensor = t.tensor(y)  # \in (1)

    def __len__(self) -> int:
        return self._y.shape[0]

    def len_vocab(self) -> int:
        return len(self._chars)

    def __getitem__(self, idx: int) -> tuple[t.Tensor, t.Tensor]:
        return self._X[idx], self._y[idx]

    def char_at(self, idx: int) -> str:
        return self._chars[idx]

    def idx_of(self, char: str) -> int:
        return self._idxs[char]
