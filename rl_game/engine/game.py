from abc import ABC, abstractmethod
from typing import Optional
import os
import sys
import threading

import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback

from rl_game.engine.handler import PlayerHandler
from rl_game.engine.registry import SobjectRegistry


class GracefulStopCallback(BaseCallback):
    """Callback that checks a threading.Event and terminates SB3 training safely."""

    def __init__(self, stop_event: threading.Event, verbose: int = 0):
        super().__init__(verbose)
        self.stop_event = stop_event

    def _on_step(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            return False  # Signals SB3 to finish learn() gracefully
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

        # Threading and synchronization
        self.stop_event = threading.Event()
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
        self.model = model
        callback = GracefulStopCallback(self.stop_event)
        try:
            self.model.learn(total_timesteps, callback=callback)
        finally:
            save_dir = os.path.dirname(model_save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            self.model.save(model_save_path)
            print(f"Model saved to {model_save_path}. Background worker exiting.")

    def run(self, model=None, total_timesteps: int = 10_000_000, model_save_path: str = "model/test"):
        if self.mode == "ai" and model is not None:
            self.stop_event.clear()
            self.worker_thread = threading.Thread(
                target=self.ai_work,
                daemon=False,
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

            if self.player and isinstance(self.player, PlayerHandler):
                if self.mode == "player":
                    self.player.handle_input(event, self.field)

    def draw_text(self, text, font, x, y):
        sur = font.render(text, True, (255,255,255))
        text_rect = sur.get_rect()
        text_rect.topleft = (x, y)
        self.screen.blit(sur, text_rect)

    @abstractmethod
    def update(self):
        pass

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.field.draw(self.screen)