from time import sleep
from typing import SupportsFloat, Any

import torch.nn as nn
from rl_game.engine.game import GameEngine
import numpy as np
from gymnasium import Env, spaces, Space
from gymnasium.core import ActType, ObsType
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from abc import ABC, abstractmethod

class AIPlayer(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim):
        super().__init__(observation_space, features_dim=features_dim)

        self.cnn =  nn.Sequential(
            nn.LazyConv2d(out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.out = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        cnn = self.cnn(x)
        return self.out(cnn)

class DQNTrainer(Env, ABC):
    def __init__(self, game: GameEngine, action_space, max_id):
        self.game: GameEngine = game
        self.player = game.player
        self.max_id = max_id

        self.action_space = spaces.Discrete(action_space)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.max_id, game.field.height, game.field.width)
        )
        self.step_count = 0



    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        :param action:
        :return:
        obs : ゲームの状態空間
        reward : 報酬
        terminated : 目的を達成しているか(bool)
        truncated : 時間制限か(bool)
        info : デバッグ情報
        """
        sleep(0.1)
        self.step_count += 1
        self.player.handle_AI_input(action, self.game.field)
        obs = self.game.field.get()
        obs = self._one_hot_encode(obs)
        base_reward = self.game.field.score_board['reward']

        master_reward, truncated, terminated = self.reward()
        reward = base_reward + master_reward
        self.game.field.score_board['reward'] = 0

        if terminated:
            self.goal()

        if truncated:
            reward -= 100

        return obs, reward, terminated, truncated, {}

    def _one_hot_encode(self, grid):
        # np.eyeは単位行列を作る関数。これを利用すると高速にone-hotエンコードできる
        # (height, width, channels) の形式で出力される
        encoded_grid = np.eye(self.max_id, dtype=np.uint8)[grid]

        # PyTorchのCNNが期待する (channels, height, width) に軸を入れ替える
        return np.transpose(encoded_grid, (2, 0, 1))

    @abstractmethod
    def goal(self):
        pass

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.player = self.game.reset()

        obs = self.game.field.get()
        obs = self._one_hot_encode(obs)
        self.goal_score = 0
        self.step_count = 0
        return obs, {}

    @abstractmethod
    def reward(self):
        pass
