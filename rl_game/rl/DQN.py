"""Backward compatibility module for rl_game.rl.DQN.

Imports and aliases classes from the refactored rl_game.rl package.
"""
from typing import Any, SupportsFloat, Tuple
from abc import ABC, abstractmethod
import numpy as np
from gymnasium import Env, spaces
from gymnasium.core import ActType, ObsType
from stable_baselines3 import DQN

from rl_game.engine.game import GameEngine
from rl_game.rl.environment import GridRLEnv
from rl_game.rl.features_extractor import AIPlayer


class DQNTrainer(GridRLEnv, ABC):
    """Legacy DQNTrainer class maintaining backward compatibility for existing implementations."""

    def __init__(self, game: GameEngine, action_space: int, max_id: int):
        super().__init__(game=game, config=action_space, max_id=max_id, step_delay=0.1)


__all__ = [
    "AIPlayer",
    "DQNTrainer",
    "DQN",
]
