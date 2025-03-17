from enum import Enum
from typing import Iterable, NamedTuple, Literal

import gymnasium as gym
import numpy as np

Position = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]] | tuple[int, int]


class Action(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class EnvState(NamedTuple):
    next_state: Position
    reward: int
    terminated: bool
    truncated: bool
    info: dict


class GridWorld(gym.Env):
    def __init__(self, size: Position, terminal_positions: Iterable[Position]):
        self._size = np.array(size)
        self._terminal_positions = np.array(terminal_positions)

        self._agent_position: Position | None = None

        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self._size[0] - 1, self._size[1] - 1])
        )
        self.action_space = gym.spaces.Discrete(4)

        self._action = {
            Action.RIGHT: np.array([0, 1]),
            Action.UP: np.array([-1, 0]),
            Action.LEFT: np.array([0, -1]),
            Action.DOWN: np.array([1, 0])
        }

    def step(self, action: Action | np.int64) -> EnvState:
        direction = self._action[action]
        self._agent_position = np.clip(
            self._agent_position + direction,
            a_min=[0, 0],
            a_max=self._size - 1
        )
        terminated = any(
            np.array_equal(self._agent_position, terminal_position)
            for terminal_position in self._terminal_positions
        )
        truncated = False
        reward = 0 if terminated else -1
        return EnvState(self._agent_position, reward, terminated, truncated, {})

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Position, dict]:
        # This will seed numpy, no need to do it separately
        # Env has an attribute np_random: np.random_generator.Generator
        super().reset(seed=seed)
        # noinspection PyTypeChecker
        self._agent_position: Position = self.np_random.integers(low=[0, 0], high=self._size, dtype=np.int32)
        return self._agent_position, {}
