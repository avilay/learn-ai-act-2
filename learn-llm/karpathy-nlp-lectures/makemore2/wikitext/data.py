from pathlib import Path

import numpy as np
import torch as t
from torch.utils.data import Dataset

EOT_TOKEN = "<0>"


class Vocab:
    def __init__(self) -> None:
        self._words: list[str] = [EOT_TOKEN]
        self._idxs: dict[str, int] = {EOT_TOKEN: 0}

    def add(self, word: str) -> None:
        if word in self._idxs:
            return
        self._words.append(word)
        self._idxs[word] = len(self._words) - 1

    def idx_of(self, word: str) -> int:
        return self._idxs.get(word, -1)

    def word_at(self, idx: int) -> str | None:
        return self._words[idx] if idx < len(self._words) else None

    def __len__(self) -> int:
        return len(self._words)


class WikitextDataset(Dataset):
    def __init__(self, filepath: Path, context_len: int) -> None:
        allwords: list[str] = self._parse(filepath)

        self.vocab = Vocab()
        for word in allwords:
            self.vocab.add(word)

        X: list[list[int]] = []
        y: list[int] = []
        context = [self.vocab.idx_of(EOT_TOKEN)] * context_len
        for word in allwords:
            X.append(context)
            idx = self.vocab.idx_of(word)
            y.append(idx)

            if word == EOT_TOKEN:
                context = [self.vocab.idx_of(EOT_TOKEN)] * context_len
            else:
                context = context[1:] + [idx]

        self._X: t.Tensor = t.tensor(X)
        self._y: t.Tensor = t.tensor(y)

    def _parse(self, filepath: Path) -> list[str]:
        allwords: list[str] = []
        with open(filepath, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    line += " " + EOT_TOKEN
                    for word in line.split():
                        allwords.append(word)
        return allwords

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int) -> tuple[t.Tensor, t.Tensor]:
        return self._X[idx], self._y[idx]

    def debug_view(self, a: t.Tensor) -> np.ndarray:
        words: list[str | None] = []
        for idx in a.flatten():
            words.append(self.vocab.word_at(idx))
        return np.array(words).reshape(a.shape)
