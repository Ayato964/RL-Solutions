from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
import os
import sys
import threading

import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback

from rl_game.engine.handler import PlayerHandler
from rl_game.engine.registry import SobjectRegistry


class GamePhase(str, Enum):
    """Execution lifecycle phase of the game engine."""
    PLAYER = "PLAYER"               # Pure manual gameplay mode (mode="player")
    HUMAN_DEMO = "HUMAN_DEMO"       # Collecting human expert demonstration transitions
    PRETRAINING = "PRETRAINING"     # Performing Behavioral Cloning pre-training
    RL_TRAINING = "RL_TRAINING"     # Reinforcement learning model training and inference
    EVALUATION = "EVALUATION"       # Post-training continuous autonomous evaluation demo mode



class GracefulStopCallback(BaseCallback):
    """Callback that checks stop_event and transition_event to terminate SB3 training safely."""

    def __init__(self, stop_event: threading.Event, transition_event: Optional[threading.Event] = None, verbose: int = 0):
        super().__init__(verbose)
        self.stop_event = stop_event
        self.transition_event = transition_event

    def _on_step(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            return False  # Signals SB3 to finish learn() gracefully
        if self.transition_event is not None and self.transition_event.is_set():
            return False  # Signals SB3 to finish learn() for mode transition
        return True


class Field:
    def __init__(self, width: int, height: int, cell_size: int, registry: SobjectRegistry):
        self.lock = threading.RLock()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.registry = registry
        self.grid = [[self.registry.create(0, x, y, self.cell_size) for x in range(width)] for y in range(height)]
        self.score_board = {
            "reward": 0,
            "score": 0,
            "apple_count": 0,
            "apple_max": 10,
            "status": "INFERENCE",
            "step": 0,
            "max_step": 1000,
            "total_step": 0,
        }

    def re_setting_sobject(self, instance, *args, **kwargs):
        with self.lock:
            for y in range(self.height):
                for x in range(self.width):
                    if isinstance(self.grid[y][x], instance):
                        self.grid[y][x].setting(*args, **kwargs)

    def set_object(self, x: int, y: int, sobject_instance):
        with self.lock:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x] = sobject_instance

    def get_sobject_count(self, target_instance) -> int:
        with self.lock:
            count = 0
            for co in self.grid:
                for sob in co:
                    if isinstance(sob, target_instance):
                        count += 1
            return count

    def print_score(self):
        with self.lock:
            print(self.score_board)

    def get_object(self, x: int, y: int):
        with self.lock:
            return self.grid[y][x]

    def get_sobjects_by_name(self, sobject_class):
        with self.lock:
            ins = []
            for i in self.grid:
                for sob in i:
                    if isinstance(sob, sobject_class):
                        ins.append(sob)
            return ins

    def get(self) -> np.ndarray:
        with self.lock:
            return np.array([[self.grid[y][x].id for x in range(self.width)] for y in range(self.height)])

    def draw(self, surface):
        with self.lock:
            for row in self.grid:
                for sobj in row:
                    sobj.draw(surface)

    def reset(self):
        with self.lock:
            status = self.score_board.get("status", "INFERENCE")
            total_step = self.score_board.get("total_step", 0)
            max_step = self.score_board.get("max_step", 1000)
            clear_count = self.score_board.get("clear_count", 0)
            total_clears = self.score_board.get("total_clears", 0)
            self.grid = [[self.registry.create(0, x, y, self.cell_size) for x in range(self.width)] for y in range(self.height)]
            self.score_board = {
                "reward": 0,
                "score": 0,
                "apple_count": 0,
                "apple_max": 10,
                "status": status,
                "step": 0,
                "max_step": max_step,
                "total_step": total_step,
                "clear_count": clear_count,
                "total_clears": total_clears,
            }


class GameEngine(ABC):
    def __init__(self, width: int, height: int, cell_size: int, mode: str = 'player'):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.mode = mode
        self.episode_count = 0
        self.rng = np.random.default_rng()

        # Lifecycle phase management
        self.phase: GamePhase = GamePhase.PLAYER if mode == 'player' else GamePhase.RL_TRAINING
        self.demo_buffer: Optional[Any] = None
        self.env: Optional[Any] = None
        self.config: Optional[Any] = None
        self.current_obs: Optional[Any] = None
        self.model: Optional[Any] = None
        self.total_timesteps: int = 10_000_000
        self.model_save_path: str = "model/test"

        # Threading and synchronization
        self.stop_event = threading.Event()
        self.eval_switch_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        # --- Initialization Order (Template Method) ---
        # 1. Prepare registry
        self.registry = SobjectRegistry()
        self.register_objects(self.registry)  # Hook method

        # 2. Setup screen and field
        window_size = (width * cell_size, height * cell_size)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Grid Game Engine")

        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = pygame.display.get_wm_info().get("window")
                if hwnd:
                    ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                    ctypes.windll.user32.ShowWindow(hwnd, 5)  # SW_SHOW
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
            except Exception:
                pass

        self.field = Field(width, height, cell_size, self.registry)

        # 3. Place objects on the field
        self.player = None
        self.setup_field()  # Hook method

        self.clock = pygame.time.Clock()
        self.running = True

    def set_rng(self, rng):
        """Sets the random number generator for reproducible simulation."""
        self.rng = rng

    @abstractmethod
    def register_objects(self, registry: SobjectRegistry):
        """Override this method to register your game's Sobjects."""
        pass

    @abstractmethod
    def setup_field(self):
        """Override this method to place objects on the field."""
        pass

    # --- Core rl_game logic ---
    def set_player(self, player_instance):
        self.player = player_instance
        self.field.set_object(self.player.x, self.player.y, self.player)

    def ai_work(self, model, total_timesteps: int = 10_000_000, model_save_path: str = "model/test"):
        import time
        self.model = model

        while not self.stop_event.is_set():
            if self.phase == GamePhase.RL_TRAINING:
                self.eval_switch_event.clear()
                callback = GracefulStopCallback(self.stop_event, transition_event=self.eval_switch_event)
                try:
                    self.model.learn(total_timesteps, callback=callback)
                except Exception as e:
                    print(f"[ERROR] AI worker thread encountered exception: {e}", file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc()
                finally:
                    save_dir = os.path.dirname(model_save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    try:
                        self.model.save(model_save_path)
                        print(f"Model saved to {model_save_path}.", flush=True)
                    except Exception as save_err:
                        print(f"[ERROR] Failed to save model: {save_err}", file=sys.stderr, flush=True)

                # Transition to continuous evaluation demo mode if completed naturally
                if not self.stop_event.is_set() and not self.eval_switch_event.is_set():
                    self.phase = GamePhase.EVALUATION
                    print("[EVALUATION] Training completed! Transitioned to evaluation demo mode. Press [T] to resume training.", flush=True)

            elif self.phase == GamePhase.EVALUATION:
                try:
                    self._run_evaluation_demo(model)
                except Exception as eval_err:
                    print(f"[ERROR] Evaluation demo encountered exception: {eval_err}", file=sys.stderr, flush=True)
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

    def _run_evaluation_demo(self, model) -> None:
        """Continuously runs the trained model autonomously across all stages for demonstration viewing."""
        import numpy as np
        import torch

        is_multi_env = hasattr(self.env, "num_envs") and self.env.num_envs > 1
        num_envs = getattr(self.env, "num_envs", 1)

        # If step_delay was 0.0 (maximum training speed), set a smooth viewing delay (~30 FPS)
        if getattr(self, "step_delay", 0.0) == 0.0:
            self.step_delay = 0.03

        reset_res = self.env.reset()
        current_obses = reset_res[0] if isinstance(reset_res, tuple) else reset_res

        has_transformer = (
            hasattr(model, "actor")
            and hasattr(model.actor, "features_extractor")
            and hasattr(model.actor.features_extractor, "encode_frame")
            and hasattr(model.actor.features_extractor, "forward_tokens")
        )
        token_histories = [[] for _ in range(num_envs)] if has_transformer else None

        if hasattr(self.field, "score_board"):
            with self.field.lock:
                self.field.score_board["status"] = "COMPLETE! [Auto Demo Play]"

        while not self.stop_event.is_set() and self.phase == GamePhase.EVALUATION:
            if is_multi_env:
                actions = []
                with torch.no_grad():
                    if has_transformer:
                        obs_tensor = torch.as_tensor(np.array(current_obses), device=model.device)
                        new_tokens = model.actor.features_extractor.encode_frame(obs_tensor)
                        for i in range(num_envs):
                            token_histories[i].append(new_tokens[i:i+1])
                            tokens = torch.cat(token_histories[i], dim=1)
                            feat = model.actor.features_extractor.forward_tokens(tokens, return_all_steps=False)
                            logits = model.actor.policy_head(feat)
                            act = int(torch.argmax(logits, dim=-1).item())
                            actions.append(act)
                    else:
                        for i in range(num_envs):
                            act, _ = model.predict(current_obses[i], deterministic=True)
                            actions.append(int(act) if not isinstance(act, np.ndarray) else int(act.item()))

                step_res = self.env.step(actions)
                next_obses, _, terminateds, truncateds = step_res[0], step_res[1], step_res[2], step_res[3]

                for i in range(num_envs):
                    if terminateds[i] or truncateds[i]:
                        if hasattr(self.env, "reset_at"):
                            r_obs, _ = self.env.reset_at(i)
                            next_obses[i] = r_obs
                        if token_histories is not None:
                            token_histories[i] = []

                current_obses = next_obses
            else:
                act, _ = model.predict(current_obses, deterministic=True)
                action = int(act) if not isinstance(act, np.ndarray) else int(act.item())
                next_obs, _, term, trunc, _ = self.env.step(action)
                if term or trunc:
                    r_obs, _ = self.env.reset()
                    next_obs = r_obs
                current_obses = next_obs

    def _key_to_action(self, key: int) -> Optional[int]:
        """Maps pygame arrow keys to standard grid action IDs (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT)."""
        if key == pygame.K_UP:
            return 0
        elif key == pygame.K_DOWN:
            return 1
        elif key == pygame.K_LEFT:
            return 2
        elif key == pygame.K_RIGHT:
            return 3
        return None

    def _step_human_demo(self, action: int) -> None:
        """Executes a single human action through Gymnasium env, mirroring RL step transition."""
        if self.env is None or self.demo_buffer is None:
            return

        obs = self.current_obs
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        self.demo_buffer.add_step(obs, action, reward, terminated, truncated)
        self.current_obs = next_obs

        if hasattr(self.field, "score_board"):
            with self.field.lock:
                self.field.score_board["status"] = (
                    f"DEMO PLAY [Steps:{self.demo_buffer.total_steps} Ep:{self.demo_buffer.total_episodes}]"
                )

        if terminated or truncated:
            self.current_obs, _ = self.env.reset()

    def _transition_to_pretraining_and_rl(self) -> None:
        """Invoked when human presses SPACE key. Runs Behavioral Cloning pre-training then launches RL."""
        if self.phase != GamePhase.HUMAN_DEMO:
            return

        self.phase = GamePhase.PRETRAINING
        if hasattr(self.field, "score_board"):
            with self.field.lock:
                self.field.score_board["status"] = "PRETRAINING (Behavioral Cloning)..."

        self.draw()
        self.update()
        pygame.display.flip()

        imitation_cfg = getattr(self.config, "imitation", None)
        if self.demo_buffer is not None:
            # Commit and persist any unwritten demonstration episodes to disk
            self.demo_buffer.save_to_directory()

        if (
            self.model is not None
            and self.demo_buffer is not None
            and not self.demo_buffer.is_empty()
            and imitation_cfg is not None
            and self.demo_buffer.total_steps >= imitation_cfg.min_demos
        ):
            print(
                f"[PRETRAIN] Initiating Behavioral Cloning over {self.demo_buffer.total_steps} demonstration steps "
                f"({self.demo_buffer.total_episodes} episodes)...",
                flush=True,
            )
            from rl_game.rl.imitation import ImitationLearnerFactory
            algo = getattr(self.config, "algorithm", "GRPO")
            learner = ImitationLearnerFactory.create(algo)

            def on_epoch_progress(epoch: int, total_epochs: int, loss: float, acc: float):
                # Pump OS/Pygame events to keep Windows desktop window manager responsive
                # and prevent the "Not Responding" white-out freeze dialog
                pygame.event.pump()
                if hasattr(self.field, "score_board"):
                    with self.field.lock:
                        self.field.score_board["status"] = (
                            f"PRETRAINING [{epoch}/{total_epochs}] (Loss:{loss:.3f} Acc:{acc*100:.0f}%)"
                        )
                self.draw()
                self.update()
                pygame.display.flip()

            metrics = learner.train(self.model, self.demo_buffer, imitation_cfg, epoch_callback=on_epoch_progress)
            loss_val = metrics.get("loss", 0.0)
            acc_val = metrics.get("accuracy", 0.0) * 100
            print(f"[PRETRAIN] BC Pre-training Finished! Loss: {loss_val:.4f}, Accuracy: {acc_val:.1f}%", flush=True)

            if hasattr(self.field, "score_board"):
                with self.field.lock:
                    self.field.score_board["status"] = (
                        f"BC CLEARED (Loss:{loss_val:.3f} Acc:{acc_val:.0f}%) -> RL"
                    )
        else:
            print("[PRETRAIN] No demonstrations recorded or min_demos not reached. Skipping BC pre-training.", flush=True)
            if hasattr(self.field, "score_board"):
                with self.field.lock:
                    self.field.score_board["status"] = "STARTING RL..."

        # Flush any key inputs accumulated during pre-training to prevent input bursts
        pygame.event.clear(pygame.KEYDOWN)
        pygame.event.clear(pygame.KEYUP)

        # Clean environment and gameplay stats transition: reset score, clear count, step counters, and difficulty/max_step
        if hasattr(self.field, "score_board"):
            with self.field.lock:
                self.field.score_board["score"] = 0
                self.field.score_board["reward"] = 0
                self.field.score_board["clear_count"] = 0
                self.field.score_board["total_clears"] = 0
                self.field.score_board["step"] = 0
                self.field.score_board["total_step"] = 0
                self.field.score_board["hit_wall"] = False
                self.field.score_board["max_step"] = getattr(self.env, "initial_max_step", 1000)

        if self.env is not None:
            if hasattr(self.env, "reset_stats"):
                self.env.reset_stats()
            else:
                if hasattr(self.env, "goal_count"):
                    self.env.goal_count = 0
                if hasattr(self.env, "total_clears"):
                    self.env.total_clears = 0
                if hasattr(self.env, "step_count"):
                    self.env.step_count = 0
                if hasattr(self.env, "total_step_count"):
                    self.env.total_step_count = 0
                if hasattr(self.env, "max_step"):
                    self.env.max_step = getattr(self.env, "initial_max_step", 1000)
            self.current_obs, _ = self.env.reset()

        # Ensure episode count starts cleanly at 0 for RL training
        self.episode_count = 0

        self.draw()
        self.update()
        pygame.display.flip()

        # Transition to RL Training phase and spawn background worker with daemon=True
        self.phase = GamePhase.RL_TRAINING
        self.stop_event.clear()
        self.worker_thread = threading.Thread(
            target=self.ai_work,
            daemon=True,
            args=(self.model, self.total_timesteps, self.model_save_path),
        )
        self.worker_thread.start()

    def run(
        self,
        model=None,
        total_timesteps: int = 10_000_000,
        model_save_path: str = "model/test",
        env: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        self.model = model
        self.total_timesteps = total_timesteps
        self.model_save_path = model_save_path
        self.env = env
        self.config = config

        imitation_enabled = (
            config is not None
            and getattr(config, "imitation", None) is not None
            and config.imitation.enabled
        )

        if self.mode == "ai" and model is not None:
            if self.phase == GamePhase.EVALUATION:
                self.stop_event.clear()
                self.worker_thread = threading.Thread(
                    target=self.ai_work,
                    daemon=True,
                    args=(model, total_timesteps, model_save_path),
                )
                self.worker_thread.start()
            elif imitation_enabled and env is not None:
                from rl_game.rl.imitation import DemonstrationBuffer
                self.phase = GamePhase.HUMAN_DEMO
                imitation_cfg = getattr(config, "imitation", None)
                save_dir = getattr(imitation_cfg, "save_dir", None)
                max_eps = getattr(imitation_cfg, "max_episodes", 100)
                min_st = getattr(imitation_cfg, "min_episode_steps", 2)
                self.demo_buffer = DemonstrationBuffer(
                    save_dir=save_dir,
                    max_episodes=max_eps,
                    min_episode_steps=min_st,
                )
                self.current_obs, _ = self.env.reset()
                if hasattr(self.field, "score_board"):
                    with self.field.lock:
                        self.field.score_board["status"] = (
                            f"DEMO PLAY [Steps:{self.demo_buffer.total_steps} Ep:{self.demo_buffer.total_episodes}] (Press SPACE to Train & Start RL)"
                        )
            else:
                self.phase = GamePhase.RL_TRAINING
                self.stop_event.clear()
                self.worker_thread = threading.Thread(
                    target=self.ai_work,
                    daemon=True,
                    args=(model, total_timesteps, model_save_path),
                )
                self.worker_thread.start()



        try:
            while self.running:
                self.handle_events()
                self.draw()
                self.update()
                pygame.display.flip()
                self.clock.tick(30)
        finally:
            if self.demo_buffer is not None:
                try:
                    self.demo_buffer.save_to_directory()
                except Exception as err:
                    print(f"[DEMO PERSISTENCE WARNING] Failed to persist demonstrations on exit: {err}", file=sys.stderr, flush=True)
            self.stop_event.set()
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=3.0)
            pygame.quit()


    def reset(self):
        self.field.reset()
        self.setup_field()
        self.episode_count += 1
        return self.player

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if self.phase == GamePhase.HUMAN_DEMO and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._transition_to_pretraining_and_rl()
                else:
                    action = self._key_to_action(event.key)
                    if action is not None:
                        self._step_human_demo(action)
            elif self.mode == "player" and self.player and isinstance(self.player, PlayerHandler):
                self.player.handle_input(event, self.field)

    def draw_text(self, text, font, x, y):
        sur = font.render(text, True, (255, 255, 255))
        text_rect = sur.get_rect()
        text_rect.topleft = (x, y)
        self.screen.blit(sur, text_rect)

    @abstractmethod
    def update(self):
        pass

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.field.draw(self.screen)