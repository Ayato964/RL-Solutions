from abc import ABC, abstractmethod
from time import sleep
from typing import Any, Optional, SupportsFloat, Tuple, Union

import numpy as np
from gymnasium import Env, spaces
from gymnasium.core import ActType, ObsType

from rl_game.engine.game import GameEngine
from rl_game.rl.config import RLConfig, TemporalConfig
from rl_game.rl.temporal import TemporalObservationBuffer


class GridRLEnv(Env, ABC):
    """Abstract base Reinforcement Learning Environment for Grid GameEngines.

    Conforms strictly to Gymnasium specifications and integrates seamlessly
    with time-series / temporal observation buffers.
    """

    def __init__(
        self,
        game: GameEngine,
        config: Optional[Union[RLConfig, int]] = None,
        max_id: Optional[int] = None,
        temporal_config: Optional[TemporalConfig] = None,
        step_delay: float = 0.05,
    ):
        super().__init__()
        self.game: GameEngine = game
        self.player = game.player

        # Allow flexible initialization either via unified RLConfig or legacy arguments
        if isinstance(config, RLConfig):
            self.action_space_dim = config.action_space
            self.max_id = config.max_id
            self.step_delay = config.step_delay
            self.temporal_config = config.temporal
        else:
            self.action_space_dim = config if isinstance(config, int) else 4
            self.max_id = max_id if max_id is not None else 4
            self.step_delay = step_delay
            self.temporal_config = temporal_config or TemporalConfig(enabled=False, n_frames=1)

        self.action_space = spaces.Discrete(self.action_space_dim)

        # Determine whether full-episode transformer sequence mode is active
        from rl_game.rl.config import TemporalType
        is_full_episode_transformer = (
            self.temporal_config.enabled
            and getattr(self.temporal_config, "temporal_type", None) == TemporalType.TRANSFORMER
            and getattr(self.temporal_config, "full_episode", True)
        )

        # Calculate channels: (n_frames * max_id) if sliding window temporal enabled, else max_id
        if self.temporal_config.enabled and not is_full_episode_transformer:
            n_frames = self.temporal_config.n_frames
        else:
            n_frames = 1
        total_channels = n_frames * self.max_id

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(total_channels, game.field.height, game.field.width),
            dtype=np.uint8,
        )

        self.temporal_buffer = TemporalObservationBuffer(
            n_frames=n_frames,
            enabled=self.temporal_config.enabled and not is_full_episode_transformer,
        )
        self.step_count = 0
        self.total_step_count = 0

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Executes a single environment step.

        Returns:
            obs: Temporal or single-frame observation.
            reward: Scalar reward for this transition.
            terminated: Whether the episode reached a terminal state (goal/death).
            truncated: Whether the episode was truncated due to time limits.
            info: Diagnostic information dictionary.
        """
        # Skip artificial sleep delay during human demonstration mode to eliminate input lag
        is_human_demo = getattr(getattr(self.game, "phase", None), "value", str(getattr(self.game, "phase", ""))) == "HUMAN_DEMO"
        if self.step_delay > 0 and not is_human_demo:
            sleep(self.step_delay)



        self.step_count += 1
        self.total_step_count += 1
        self.player.handle_AI_input(action, self.game.field)

        # Dynamic game object replenishment (e.g., apples)
        if hasattr(self.game, "replenish_apples"):
            self.game.replenish_apples()

        raw_grid = self.game.field.get()
        frame = self._one_hot_encode(raw_grid)
        obs = self.temporal_buffer.append(frame)

        base_reward = self.game.field.score_board.get("reward", 0)
        master_reward, truncated, terminated = self.reward()
        reward = float(base_reward + master_reward)
        self.game.field.score_board["reward"] = 0

        # Update step counters for UI HUD display
        if hasattr(self.game, "field") and hasattr(self.game.field, "score_board"):
            with self.game.field.lock:
                self.game.field.score_board["step"] = self.step_count
                self.game.field.score_board["total_step"] = self.total_step_count

        if terminated:
            self.goal()

        return obs, reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[ObsType, dict[str, Any]]:
        """Resets the game field and populates the temporal observation buffer."""
        super().reset(seed=seed)
        if hasattr(self.game, "set_rng"):
            self.game.set_rng(self.np_random)

        self.player = self.game.reset()

        raw_grid = self.game.field.get()
        initial_frame = self._one_hot_encode(raw_grid)
        obs = self.temporal_buffer.reset(initial_frame)

        self.step_count = 0
        if hasattr(self.game, "field") and hasattr(self.game.field, "score_board"):
            with self.game.field.lock:
                self.game.field.score_board["step"] = 0
        return obs, {}

    def _one_hot_encode(self, grid: np.ndarray) -> np.ndarray:
        """Encodes 2D integer grid into (channels, height, width) one-hot array."""
        encoded = np.eye(self.max_id, dtype=np.uint8)[grid]
        return np.transpose(encoded, (2, 0, 1))

    @abstractmethod
    def reward(self) -> Tuple[float, bool, bool]:
        """Calculates (reward, truncated, terminated)."""
        pass

    @abstractmethod
    def goal(self) -> None:
        """Callback invoked upon reaching a terminal goal condition."""
        pass
