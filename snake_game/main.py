import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from typing import Any, Optional, Tuple, Union

import pygame
from gymnasium.core import ObsType

from rl_game.engine.game import GameEngine
from rl_game.engine.registry import SobjectRegistry, Void
from rl_game.rl import (
    AlgorithmType,
    DQNConfig,
    GRPOConfig,
    GridRLEnv,
    PPOConfig,
    RLConfig,
    RLModelFactory,
    TemporalConfig,
    TemporalType,
)
try:
    from snake_game.sobjects import Register, Wall
except ImportError:
    from sobjects import Register, Wall


class BadAppleEnv(GridRLEnv):
    """Gymnasium Environment for BadApple Snake game.

    Integrates with GridRLEnv to support temporal observation windowing (LSTM, GRU, Conv1D, Stack)
    and pluggable RL algorithms (PPO, DQN).
    """

    def __init__(
        self,
        game: GameEngine,
        config: Optional[Union[RLConfig, int]] = None,
        max_id: Optional[int] = None,
        terminate_on_wall: bool = True,
        step_penalty: float = 0.01,
        wall_penalty: float = 50.0,
        **kwargs,
    ):
        super().__init__(game=game, config=config, max_id=max_id, **kwargs)
        self.max_step = 1000
        self.goal_count = 0
        self.total_clears = 0
        self.is_goal = False
        self.terminate_on_wall = terminate_on_wall
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        if hasattr(self.game, "field"):
            self.game.field.wall_penalty = wall_penalty
            if hasattr(self.game.field, "score_board"):
                with self.game.field.lock:
                    self.game.field.score_board["max_step"] = self.max_step
                    self.game.field.score_board["clear_count"] = self.goal_count
                    self.game.field.score_board["total_clears"] = self.total_clears

    def reward(self) -> Tuple[float, bool, bool]:
        """Calculates step reward, truncated status (time-over), and terminated status (game-over).

        Returns:
            (reward, truncated, terminated)
        """
        score = self.game.field.score_board.get("score", 0)
        goal_score = 500 if score >= 1000 else 0
        hit_wall = self.game.field.score_board.get("hit_wall", False)

        # タイムオーバー (Time-Over): 最大ステップ数到達
        truncated = self.step_count >= self.max_step
        # ゲームオーバー (Game-Over): 目標スコア達成 または 壁衝突
        terminated = (score >= 1000) or (self.terminate_on_wall and hit_wall)

        return -self.step_penalty + goal_score, truncated, terminated

    def goal(self) -> None:
        """Invoked when an episode terminates; handles game clear (score >= 1000) and progressive difficulty."""
        score = self.game.field.score_board.get("score", 0)
        # 壁衝突やタイムオーバー等、ゲームクリア(score >= 1000)以外の終了時は何もしない
        if score < 1000:
            return

        self.goal_count += 1
        self.total_clears += 1
        self.is_goal = True
        print(f"[CLEAR] Game Cleared! Progress: {self.goal_count}/10 clears (Total Clears: {self.total_clears})", flush=True)

        if self.goal_count >= 10:
            self.max_step = max(100, self.max_step - 100)
            self.goal_count = 0
            print(f"----------------- LEVEL UP! max_step reduced to {self.max_step} --------------------------", flush=True)

        if hasattr(self.game, "field") and hasattr(self.game.field, "score_board"):
            with self.game.field.lock:
                self.game.field.score_board["max_step"] = self.max_step
                self.game.field.score_board["clear_count"] = self.goal_count
                self.game.field.score_board["total_clears"] = self.total_clears

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        if hasattr(self.game, "field"):
            self.game.field.wall_penalty = self.wall_penalty
        if self.is_goal:
            # Change wall colors deterministically using seeded game RNG
            new_color = (
                int(self.game.rng.integers(50, 256)),
                int(self.game.rng.integers(50, 256)),
                int(self.game.rng.integers(50, 256)),
            )
            self.game.field.re_setting_sobject(Wall, color=new_color)
            self.is_goal = False
            print("RESET with new wall colors")
        return obs, info


# Backward compatibility alias
BadAppleTrainer = BadAppleEnv


class BadAppleGame(GameEngine):
    """Pygame-based Grid Game for BadApple RL training and rendering."""

    def __init__(self, width: int = 30, height: int = 20, cell_size: int = 35):
        super().__init__(width, height, cell_size, mode="ai")
        self.font = pygame.font.SysFont("", 32)

    def register_objects(self, registry: SobjectRegistry) -> None:
        registry.registers(Register())

    def setup_field(self) -> None:
        player = self.registry.create(id=1, x=5, y=3, size=self.cell_size)
        self.set_player(player)

        # Outer perimeter walls
        for x in range(self.field.width):
            top_wall = self.registry.create(id=2, x=x, y=0, size=self.cell_size)
            bottom_wall = self.registry.create(id=2, x=x, y=self.field.height - 1, size=self.cell_size)
            self.field.set_object(x, 0, top_wall)
            self.field.set_object(x, self.field.height - 1, bottom_wall)

        for y in range(1, self.field.height - 1):
            left_wall = self.registry.create(id=2, x=0, y=y, size=self.cell_size)
            right_wall = self.registry.create(id=2, x=self.field.width - 1, y=y, size=self.cell_size)
            self.field.set_object(0, y, left_wall)
            self.field.set_object(self.field.width - 1, y, right_wall)

        # Inner obstacle walls
        inner_walls = [
            (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12),
            (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (20, 10), (20, 11), (20, 12),
        ]
        for x, y in inner_walls:
            wall_instance = self.registry.create(id=2, x=x, y=y, size=self.cell_size)
            self.field.set_object(x, y, wall_instance)

        self.field.score_board["apple_count"] = 0
        self.field.score_board["apple_max"] = 10
        self.replenish_apples()

    def update(self) -> None:
        self.replenish_apples()

        status = self.field.score_board.get("status", "INFERENCE")
        score = self.field.score_board.get("score", 0)
        reward = self.field.score_board.get("reward", 0)
        step = self.field.score_board.get("step", 0)
        max_step = self.field.score_board.get("max_step", 1000)
        total_step = self.field.score_board.get("total_step", step)
        clear_count = self.field.score_board.get("clear_count", 0)

        score_text = (
            f"SCORE:{score}/1000  "
            f"Ep:{self.episode_count}  "
            f"Clear:{clear_count}/10  "
            f"Step:{step}/{max_step} (Total:{total_step})  "
            f"[{status}]"
        )
        self.draw_text(score_text, self.font, 20, 20)

    def add_apple(self) -> bool:
        """Spawns an apple in an available Void cell using seeded RNG.

        Returns True if an apple was placed, False if no empty cell is available.
        """
        with self.field.lock:
            empty_cells = []
            for y in range(self.height):
                for x in range(self.width):
                    if isinstance(self.field.get_object(x, y), Void):
                        empty_cells.append((x, y))

            if not empty_cells:
                return False

            idx = int(self.rng.integers(0, len(empty_cells)))
            x, y = empty_cells[idx]
            apple = self.registry.create(id=3, x=x, y=y, size=self.cell_size)
            self.field.set_object(x, y, apple)
            self.field.score_board["apple_count"] += 1
            return True

    def replenish_apples(self) -> None:
        """Replenishes apples up to apple_max safely."""
        with self.field.lock:
            while self.field.score_board["apple_count"] < self.field.score_board["apple_max"]:
                if not self.add_apple():
                    break


def create_default_config() -> RLConfig:
    """Creates the default RL Configuration.

    To switch between algorithms, set algorithm=AlgorithmType.PPO or AlgorithmType.DQN.
    To switch temporal models, set temporal_type to LSTM, GRU, CONV1D, or STACK.
    """
    return RLConfig(
        algorithm=AlgorithmType.GRPO,  # <--- Critic-Free GRPO by default! (Switchable to PPO or DQN)
        action_space=4,
        max_id=4,
        step_delay=0.03,  # Delay per step in seconds for visual observation (0 for max training speed)
        total_timesteps=1_000_000,
        model_save_path="model/bad_apple_rl",
        temporal=TemporalConfig(
            enabled=True,
            full_episode=True,                        # Full-Episode sequence from start to game-over
            max_seq_len=1000,                         # Exactly matches maximum game steps (1000)
            n_frames=1,                               # Single frame per step; full sequence processed via Causal Transformer
            temporal_type=TemporalType.TRANSFORMER,    # Causal Transformer
            hidden_dim=128,                           # d_model for soft tokens
            features_dim=128,                         # Upgraded to 128
            spatial_channels=(64, 128),               # Upgraded to 128 channels
            n_heads=4,                                # 4-head self-attention
            n_layers=2,                               # 2-layer Transformer encoder
            dim_feedforward=256,                      # Feedforward dimension
            dropout=0.0,
            aggregation="last",                       # Head aggregation: 'last', 'mean', or 'attention'
        ),
        grpo=GRPOConfig(
            learning_rate=3e-4,
            group_size=32,                            # 32 completed episodes per batch
            rollout_steps=None,                       # None: infer continuously until Game Over (terminated) or Time Over (truncated)
            batch_size=64,
            n_epochs=1,                               # 1 epoch per rollout group to prevent ratio saturation
            gradient_accumulation_steps=None,         # Full-group gradient accumulation for 1 clean update
            advantage_mode="step_discounted",         # Step-level temporal discounted return relative advantage
            clip_range=0.2,
            kl_coef=0.0,
            ent_coef=0.01,
            tensorboard_log="./grpo_custom_log/",
        ),
        ppo=PPOConfig(
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            tensorboard_log="./ppo_custom_log/",
        ),
        dqn=DQNConfig(
            learning_rate=1e-4,
            buffer_size=50_000,
            batch_size=64,
            learning_starts=1000,
            gradient_steps=4,
            train_freq=(4, "step"),
            tensorboard_log="./dqn_custom_log/",
        ),
    )


if __name__ == "__main__":
    # 1. Initialize Game Configuration
    config = create_default_config()

    # 2. Build Game and RL Environment
    game = BadAppleGame(width=30, height=20, cell_size=35)
    env = BadAppleEnv(game=game, config=config)

    # 3. Build Model via Factory (PPO, DQN, GRPO automatically selected from config)
    model = RLModelFactory.create(config=config, env=env)

    print(f"[INIT] Initialized {config.algorithm.value} model with {config.temporal.temporal_type.value} temporal extractor.", flush=True)
    print(f"   Observation Space: {env.observation_space.shape}", flush=True)
    print(f"   Action Space: {env.action_space}", flush=True)

    # 4. Start Training & Visual Rendering Loop
    game.run(
        model=model,
        total_timesteps=config.total_timesteps,
        model_save_path=config.model_save_path,
    )