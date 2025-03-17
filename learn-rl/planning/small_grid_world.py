from collections import defaultdict
from itertools import product
from typing import Callable, Iterable, Literal, NamedTuple

import numpy as np


def argmax(
    vals: Iterable[float], key: Callable[[float], float] = lambda x: x
) -> Iterable[float]:
    kvals = defaultdict(list)
    for val in vals:
        kval = key(val)
        kvals[kval].append(val)
    return kvals[max(kvals)]


# UP = np.array([-1, 0])
# DOWN = np.array([1, 0])
# LEFT = np.array([0, -1])
# RIGHT = np.array([0, 1])

# State = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]
# Action = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]


class State(NamedTuple):
    row: int
    col: int

    def to_numpy(self) -> np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]:
        return np.array([self.row, self.col])

    def __repr__(self):
        return f"({self.row}, {self.col})"


class Action(NamedTuple):
    row: int
    col: int
    name: str

    @classmethod
    def up(cls) -> "Action":
        return cls(-1, 0, "↑")

    @classmethod
    def down(cls) -> "Action":
        return cls(1, 0, "↓")

    @classmethod
    def left(cls) -> "Action":
        return cls(0, -1, "←")

    @classmethod
    def right(cls) -> "Action":
        return cls(0, 1, "→")

    def to_numpy(self) -> np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]:
        return np.array([self.row, self.col])

    def __repr__(self):
        return self.name


Policy = Callable[[Action, State], float]
Grid = np.ndarray[tuple[Literal[4]], np.dtype[np.int32]]


class Grid:
    def __init__(self, size=4, random=False):
        self._grid = np.zeros((size, size), dtype=np.float32)
        if random:
            rng = np.random.default_rng()
            self._grid = rng.uniform(0.0, 0.1, (size, size))

    def __getitem__(self, key: State) -> float:
        return self._grid[key.row, key.col]

    def __setitem__(self, key: State, value: float):
        self._grid[key.row, key.col] = value

    def copy(self, other: Grid):
        self._grid = other._grid.copy()

    def clear(self):
        self._grid.fill(0)

    def __repr__(self):
        return self._grid.__repr__()

    def __eq__(self, other):
        return np.array_equal(self._grid, other._grid)

    def close(self, other):
        return np.allclose(self._grid, other._grid)

    def round(self):
        self._grid = np.rint(self._grid)


class SmallGridWorld:
    def __init__(self, gamma=1.0):
        self._gamma = gamma
        # self._terminal_states = [np.array([0, 0]), np.array([3, 3])]
        self._terminal_states = [State(0, 0), State(3, 3)]

    @property
    def gamma(self) -> float:
        return self._gamma

    def is_terminal(self, state: State) -> bool:
        return state in self._terminal_states

    def states(self) -> Iterable[State]:
        return [State(i, j) for i, j in product(range(4), range(4))]

    def actions(self) -> Iterable[Action]:
        return [Action.up(), Action.down(), Action.left(), Action.right()]

    def reward(self, state, action):
        return 0 if self.is_terminal(state) else -1

    def prob(self, next_state: State, given: tuple[State, Action]) -> float:
        state, action = given
        if self.is_terminal(state):
            return 0.0
        expected_next_state = state.to_numpy() + action.to_numpy()
        expected_next_state = np.clip(expected_next_state, a_min=[0, 0], a_max=[3, 3])
        return (
            1.0 if np.array_equal(expected_next_state, next_state.to_numpy()) else 0.0
        )
