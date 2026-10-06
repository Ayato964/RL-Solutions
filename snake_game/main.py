import os
import sys
# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from time import sleep
from typing import Any, List, Optional, Tuple, Union, overload

import numpy as np
import pygame
from gymnasium.core import ObsType

from rl_game.engine.game import Field, GameEngine, GamePhase
from rl_game.engine.registry import SobjectRegistry, Void
from rl_game.rl import (
    AlgorithmType,
    DQNConfig,
    GRPOConfig,
    GridRLEnv,
    ImitationConfig,
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
        step_penalty: float = 0.0,
        survival_reward: Optional[float] = None,
        initial_wall_penalty: float = 100.0,
        wall_penalty_decay: float = 0.1,
        wall_penalty: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(game=game, config=config, max_id=max_id, **kwargs)
        self.initial_max_step = 1000
        self.max_step = self.initial_max_step
        self.goal_count = 0
        self.total_clears = 0
        self.is_goal = False
        self.terminate_on_wall = terminate_on_wall
        self.step_penalty = step_penalty
        if survival_reward is not None:
            self.survival_reward = survival_reward
        elif isinstance(config, RLConfig) and hasattr(config, "survival_reward"):
            self.survival_reward = config.survival_reward
        else:
            self.survival_reward = 0.5
        self.initial_wall_penalty = initial_wall_penalty if wall_penalty is None else wall_penalty
        self.wall_penalty_decay = wall_penalty_decay
        self.wall_penalty = self.initial_wall_penalty
        if hasattr(self.game, "field"):
            self.game.field.wall_penalty = self.initial_wall_penalty
            if hasattr(self.game.field, "score_board"):
                with self.game.field.lock:
                    self.game.field.score_board["max_step"] = self.max_step
                    self.game.field.score_board["clear_count"] = self.goal_count
                    self.game.field.score_board["total_clears"] = self.total_clears

    def reset_stats(self) -> None:
        """Resets all gameplay progress, difficulty/max_step, and step counters back to initial baseline."""
        self.goal_count = 0
        self.total_clears = 0
        self.is_goal = False
        self.max_step = self.initial_max_step
        self.step_count = 0
        self.total_step_count = 0
        self.wall_penalty = self.initial_wall_penalty
        if hasattr(self.game, "field"):
            self.game.field.wall_penalty = self.initial_wall_penalty
            if hasattr(self.game.field, "score_board"):
                with self.game.field.lock:
                    self.game.field.score_board["max_step"] = self.max_step
                    self.game.field.score_board["clear_count"] = 0
                    self.game.field.score_board["total_clears"] = 0
                    self.game.field.score_board["score"] = 0
                    self.game.field.score_board["reward"] = 0
                    self.game.field.score_board["step"] = 0
                    self.game.field.score_board["total_step"] = 0
                    self.game.field.score_board["hit_wall"] = False

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict[str, Any]]:
        """Executes a single step with dynamically decaying wall collision penalty based on survival steps."""
        # Calculate wall collision penalty for current survival step count before step increment
        steps_alive = max(0, self.step_count)
        current_wall_penalty = max(0.0, self.initial_wall_penalty - self.wall_penalty_decay * steps_alive)
        self.wall_penalty = current_wall_penalty
        if hasattr(self.game, "field"):
            self.game.field.wall_penalty = current_wall_penalty

        return super().step(action)

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

        # 生存報酬 (Survival Bonus): 壁衝突で死亡していない安全な生存ステップに付与
        step_reward = float(goal_score - self.step_penalty)
        if not (self.terminate_on_wall and hit_wall):
            step_reward += self.survival_reward

        return step_reward, truncated, terminated

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
        self.wall_penalty = self.initial_wall_penalty
        if hasattr(self.game, "field"):
            self.game.field.wall_penalty = self.initial_wall_penalty
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


class StageEngine:
    """Lightweight individual stage controller for parallel environments."""

    def __init__(self, parent_game: "BadAppleGame", env_id: int):
        self.parent_game = parent_game
        self.env_id = env_id
        self.registry = parent_game.registry
        self.width = parent_game.width
        self.height = parent_game.height
        self.cell_size = parent_game.cell_size
        self.rng = np.random.default_rng(seed=42 + env_id)
        self.field = Field(self.width, self.height, self.cell_size, self.registry)
        self.player = None
        self.setup_field()

    def set_player(self, player_instance):
        self.player = player_instance
        self.field.set_object(self.player.x, self.player.y, self.player)

    def set_rng(self, rng):
        self.rng = rng

    def setup_field(self):
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
            if x < self.field.width and y < self.field.height:
                wall_instance = self.registry.create(id=2, x=x, y=y, size=self.cell_size)
                self.field.set_object(x, y, wall_instance)

        self.field.score_board["apple_count"] = 0
        self.field.score_board["apple_max"] = 10
        self.replenish_apples()

    def add_apple(self) -> bool:
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
        with self.field.lock:
            while self.field.score_board["apple_count"] < self.field.score_board["apple_max"]:
                if not self.add_apple():
                    break

    def reset(self):
        self.field.reset()
        self.setup_field()
        return self.player

    @property
    def phase(self):
        return self.parent_game.phase


class MultiBadAppleEnv:
    """Vectorized Environment container managing 16 parallel BadAppleEnv instances."""

    def __init__(
        self,
        game: "BadAppleGame",
        config: Optional[Union[RLConfig, int]] = None,
        num_envs: int = 16,
        **kwargs,
    ):
        self.game = game
        self.config = config
        self.num_envs = num_envs
        self.envs = [
            BadAppleEnv(
                game=game.stage_engines[i],
                config=config,
                **kwargs,
            )
            for i in range(num_envs)
        ]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        is_human_demo = (getattr(self.game, "phase", None) == GamePhase.HUMAN_DEMO)
        if is_human_demo:
            obs, info = self.envs[0].reset(seed=seed, options=options)
            return obs, info

        obses = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=None if seed is None else seed + i, options=options)
            obses.append(obs)
        return np.array(obses), {}

    def reset_at(self, idx: int, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        return self.envs[idx].reset(seed=seed, options=options)

    def step_demo(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Executes a single human demonstration step on primary stage (Stage 0)."""
        return self.envs[0].step(action)

    @overload
    def step(self, actions: int) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...

    @overload
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[dict[str, Any]]]: ...

    def step(self, actions: Union[List[int], int]):
        if isinstance(actions, (int, np.integer)):
            # Human demo single-action on primary stage (Stage 0)
            return self.step_demo(int(actions))

        delay = getattr(self.game, "step_delay", 0.0)
        is_human_demo = (getattr(self.game, "phase", None) == GamePhase.HUMAN_DEMO)
        if delay > 0 and not is_human_demo:
            pygame.time.delay(int(delay * 1000))

        obses = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        for i, env in enumerate(self.envs):
            act = actions[i] if i < len(actions) else 0
            obs, r, term, trunc, info = env.step(act)
            obses.append(obs)
            rewards.append(r)
            terminateds.append(term)
            truncateds.append(trunc)
            infos.append(info)
        return obses, rewards, terminateds, truncateds, infos

    def reset_stats(self) -> None:
        for env in self.envs:
            env.reset_stats()


class BadAppleGame(GameEngine):
    """Pygame-based Grid Game for BadApple RL training supporting 16-stage parallel rendering."""

    def __init__(self, width: int = 30, height: int = 20, cell_size: int = 12, num_envs: int = 16):
        self.num_envs = num_envs
        self.cols = 4 if num_envs == 16 else 1
        self.rows = 4 if num_envs == 16 else 1
        self.stage_w = width * cell_size
        self.stage_h = height * cell_size
        self.hud_h = 60

        super().__init__(width, height, cell_size, mode="ai")

        # Re-initialize screen with 16-stage grid layout if num_envs > 1
        if num_envs > 1:
            total_w = self.stage_w * self.cols
            total_h = self.hud_h + self.stage_h * self.rows
            self.screen = pygame.display.set_mode((total_w, total_h))
            pygame.display.set_caption("Bad Apple - 16 Parallel RL Stages")

        self.font = pygame.font.SysFont("", 24)
        self.sub_font = pygame.font.SysFont("", 18)
        self.step_delay = 0.0  # Default to max speed (0ms delay)

        # Build StageEngines for parallel environments
        self.stage_engines = [StageEngine(self, i) for i in range(num_envs)]
        # Map primary field and player to stage 0 for backward compatibility
        self.field = self.stage_engines[0].field
        self.player = self.stage_engines[0].player

        # Pre-allocate surfaces for fast blitting
        self.stage_surfaces = [pygame.Surface((self.stage_w, self.stage_h)) for _ in range(num_envs)]

        # Pre-render static stage number and HUD labels to eliminate per-frame text rasterization & GC churn
        self.stage_labels = [self.sub_font.render(f"#{i+1}", True, (120, 120, 120)) for i in range(num_envs)]
        self.demo_label = self.sub_font.render("PLAY HERE", True, (255, 215, 0))

    def register_objects(self, registry: SobjectRegistry) -> None:
        registry.registers(Register())

    def set_rng(self, rng):
        super().set_rng(rng)
        if hasattr(self, "stage_engines") and self.stage_engines:
            self.stage_engines[0].set_rng(rng)
            for stage in self.stage_engines[1:]:
                seed_val = int(rng.integers(0, 1_000_000_000))
                stage.set_rng(np.random.default_rng(seed_val))

    def setup_field(self) -> None:
        if hasattr(self, "stage_engines") and self.stage_engines:
            for stage in self.stage_engines:
                stage.setup_field()
            self.field = self.stage_engines[0].field
            self.player = self.stage_engines[0].player

    def reset(self):
        self.episode_count += 1
        if hasattr(self, "stage_engines") and self.stage_engines:
            for stage in self.stage_engines:
                stage.reset()
            self.field = self.stage_engines[0].field
            self.player = self.stage_engines[0].player
            return self.player
        return super().reset()

    def add_apple(self) -> bool:
        if hasattr(self, "stage_engines") and self.stage_engines:
            return self.stage_engines[0].add_apple()
        return False

    def replenish_apples(self) -> None:
        if hasattr(self, "stage_engines") and self.stage_engines:
            for stage in self.stage_engines:
                stage.replenish_apples()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                # Speed control shortcuts
                if event.key == pygame.K_LEFTBRACKET:
                    self.step_delay = min(0.1, self.step_delay + 0.005)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.step_delay = max(0.0, self.step_delay - 0.005)
                elif event.key == pygame.K_f:
                    self.step_delay = 0.0
                elif event.key == pygame.K_t:
                    # Toggle between EVALUATION demo mode and RL_TRAINING mode
                    if self.phase == GamePhase.EVALUATION:
                        self.phase = GamePhase.RL_TRAINING
                        if hasattr(self.field, "score_board"):
                            with self.field.lock:
                                self.field.score_board["status"] = "RESUMING RL TRAINING..."
                        print("\n[MODE SWITCH] Switched to RL Training mode via [T] key!", flush=True)
                    elif self.phase == GamePhase.RL_TRAINING:
                        self.eval_switch_event.set()
                        self.phase = GamePhase.EVALUATION
                        if hasattr(self.field, "score_board"):
                            with self.field.lock:
                                self.field.score_board["status"] = "SWITCHING TO EVAL DEMO..."
                        print("\n[MODE SWITCH] Pausing RL and switching to Evaluation Demo mode via [T] key!", flush=True)
                elif event.key == pygame.K_SPACE and self.phase == GamePhase.HUMAN_DEMO:
                    self._transition_to_pretraining_and_rl()
                elif self.phase == GamePhase.HUMAN_DEMO:
                    action = self._key_to_action(event.key)
                    if action is not None:
                        self._step_human_demo(action)

    def draw(self) -> None:
        self.screen.fill((20, 20, 20))

        # Render all parallel stages
        for i, stage in enumerate(self.stage_engines):
            col = i % self.cols
            row = i // self.cols
            pos_x = col * self.stage_w
            pos_y = self.hud_h + row * self.stage_h

            surf = self.stage_surfaces[i]
            surf.fill((0, 0, 0))
            stage.field.draw(surf)
            self.screen.blit(surf, (pos_x, pos_y))

            # Border separators
            pygame.draw.rect(self.screen, (60, 60, 60), (pos_x, pos_y, self.stage_w, self.stage_h), 1)

            # Stage number indicator (pre-baked cache)
            self.screen.blit(self.stage_labels[i], (pos_x + 6, pos_y + 4))

            # Highlight Stage 1 during DEMO PLAY
            if self.phase == GamePhase.HUMAN_DEMO and i == 0:
                pygame.draw.rect(self.screen, (255, 215, 0), (pos_x, pos_y, self.stage_w, self.stage_h), 2)
                self.screen.blit(self.demo_label, (pos_x + self.stage_w - 85, pos_y + 4))

    def update(self) -> None:
        for stage in self.stage_engines:
            stage.replenish_apples()

        scores = [stage.field.score_board.get("score", 0) for stage in self.stage_engines]
        avg_score = float(np.mean(scores)) if scores else 0.0
        max_score = int(np.max(scores)) if scores else 0
        total_clears = sum(stage.field.score_board.get("total_clears", 0) for stage in self.stage_engines)
        status = self.field.score_board.get("status", "INFERENCE")

        # Top line: Summary stats & speed control info
        fps_info = f"Delay:{self.step_delay*1000:.0f}ms (Keys: [ / ] / F)"
        line1 = (
            f"[16-PARALLEL] Score Avg:{avg_score:.1f} Max:{max_score} | "
            f"Clears:{total_clears} | Ep:{self.episode_count} | {fps_info}"
        )
        self.draw_text(line1, self.font, 20, 8)

        # Sub line: Phase status and guidance
        if self.phase == GamePhase.HUMAN_DEMO:
            line2 = f"[{status}]  -->  Play on Stage #1. Press [SPACE] to Train & Start RL"
        elif self.phase == GamePhase.EVALUATION:
            line2 = f"★ [EVALUATION DEMO] ★ Press [T] to Resume Training | Keys: [ / ] (Speed) / F (Fastest)"
        else:
            line2 = f"[{status}]  |  16 Stages (Press [T] to Pause & View Demo | Keys: [ / ] / F)"
        self.draw_text(line2, self.font, 20, 32)


def create_default_config() -> RLConfig:
    """Creates the default RL Configuration optimized for 16-parallel high-speed training."""
    human_data_dir = os.path.join(os.path.dirname(__file__), "human_data")
    os.makedirs(human_data_dir, exist_ok=True)

    return RLConfig(
        algorithm=AlgorithmType.GRPO,  # <--- Critic-Free GRPO by default! (Switchable to PPO or DQN)
        action_space=4,
        max_id=4,
        step_delay=0.0,  # 0ms delay for maximum parallel GPU/CPU throughput
        survival_reward=0.5,  # Positive reward per alive step to eliminate suicide local optimum
        total_timesteps=1_000_000,
        model_save_path="model/bad_apple_rl",
        imitation=ImitationConfig(
            enabled=True,            # Spaceキーを押すまで人間が操作してデモ収集＆模倣学習
            epochs=15,               # 模倣学習（BC）エポック数
            batch_size=32,           # ミニバッチサイズ
            learning_rate=1e-3,      # BC学習率
            min_demos=1,             # 模倣学習を行う最小ステップ数
            save_dir=human_data_dir, # 人間がプレイしたデモの永続化・再起動時マージ保存先
        ),

        temporal=TemporalConfig(
            enabled=True,
            full_episode=True,                        # Full-Episode sequence from start to game-over
            max_seq_len=1000,                         # Exactly matches maximum game steps (1000)
            n_frames=1,                               # Single frame per step; full sequence processed via Causal Transformer
            temporal_type=TemporalType.TRANSFORMER,    # Causal Transformer
            hidden_dim=256,                           # d_model for soft tokens (256-dim)
            features_dim=256,                         # Upgraded to 256-dim
            spatial_channels=(64, 128),               # Upgraded to 128 channels
            n_heads=4,                                # 4-head self-attention
            n_layers=2,                               # 2-layer Transformer encoder
            dim_feedforward=512,                      # Feedforward dimension (512-dim)
            dropout=0.0,
            aggregation="last",                       # Head aggregation: 'last', 'mean', or 'attention'
        ),
        grpo=GRPOConfig(
            learning_rate=3e-4,
            group_size=32,                            # 32 completed episodes per batch (just 2 rounds of 16-parallel)
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


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # 1. Initialize Game Configuration
    config = create_default_config()

    # 2. Build 16-Parallel Game and Multi-Environment
    num_parallel_envs = 16
    game = BadAppleGame(width=30, height=20, cell_size=12, num_envs=num_parallel_envs)
    env = MultiBadAppleEnv(game=game, config=config, num_envs=num_parallel_envs)

    # 3. Build Model via Factory (PPO, DQN, GRPO automatically selected from config)
    model = RLModelFactory.create(config=config, env=env)

    # Automatically load existing trained model weights if present
    ckpt_path = config.model_save_path if (config.model_save_path.endswith(".pt") or config.model_save_path.endswith(".zip")) else f"{config.model_save_path}.pt"
    if os.path.exists(ckpt_path):
        try:
            import torch
            checkpoint = torch.load(ckpt_path, map_location=getattr(model, "device", "cpu"))
            if hasattr(model, "actor") and "actor_state_dict" in checkpoint:
                model.actor.load_state_dict(checkpoint["actor_state_dict"])
                if hasattr(model, "ref_actor"):
                    model.ref_actor.load_state_dict(checkpoint["actor_state_dict"])
            if "num_timesteps" in checkpoint:
                model.num_timesteps = checkpoint["num_timesteps"]
            print(f"[MODEL LOAD] Successfully loaded existing checkpoint from '{ckpt_path}' ({getattr(model, 'num_timesteps', 0)} steps)!", flush=True)
        except Exception as load_err:
            print(f"[MODEL LOAD WARNING] Failed to load '{ckpt_path}': {load_err}", flush=True)

    # Check for 'eval' command-line argument to launch directly in demo evaluation mode
    is_eval_mode = any(arg.lower() in ("eval", "--eval", "-e") for arg in argv[1:])
    if is_eval_mode:
        game.phase = GamePhase.EVALUATION
        game.step_delay = 0.03
        if config.imitation is not None:
            config.imitation.enabled = False
        print("\n" + "="*70)
        print("[CLI] EVALUATION DEMO MODE ACTIVATED!")
        print("   Directly streaming autonomous AI play across 16 parallel stages.")
        print("   Controls: [ / ] = Adjust speed | F = Max speed | T = Resume Training")
        print("="*70 + "\n", flush=True)

    print(f"[INIT] Initialized {config.algorithm.value} model with {config.temporal.temporal_type.value} temporal extractor.", flush=True)
    print(f"   Parallel Environments: {num_parallel_envs} stages (4x4 grid layout)", flush=True)
    print(f"   Observation Space: {env.observation_space.shape}", flush=True)
    print(f"   Action Space: {env.action_space}", flush=True)
    print(f"   Step Delay: {config.step_delay*1000:.0f}ms (Max Speed Mode; use [ / ] / F to adjust)", flush=True)
    if config.imitation.enabled:
        print("   Imitation Pre-training: ENABLED (Play on Stage #1 with arrow keys; press SPACE to pre-train & start RL)", flush=True)
        if config.imitation.save_dir:
            print(f"   Human Demonstration Storage: {config.imitation.save_dir}", flush=True)

    # 4. Start Training & Visual Rendering Loop
    game.run(
        model=model,
        total_timesteps=config.total_timesteps,
        model_save_path=config.model_save_path,
        env=env,
        config=config,
    )


if __name__ == "__main__":
    main()