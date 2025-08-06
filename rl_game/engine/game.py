import threading

import pygame
import sys
import numpy as np

from rl_game.engine.handler import PlayerHandler
from rl_game.engine.registry import SobjectRegistry
from abc import abstractmethod, ABC

class Field:
    def __init__(self, width, height, cell_size, registry):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.registry = registry
        self.grid = [[self.registry.create(0, x, y, self.cell_size) for x in range(width)] for y in range(height)]
        self.score_board = {
            "reward": 0,
            "score": 0
        }

    def re_setting_sobject(self, instance, *args, **kwargs):
        """
        Replaces all instances of a specific Sobject with a new instance.
        :param instance: The Sobject class to replace.
        :param args: Positional arguments for the new instance.
        :param kwargs: Keyword arguments for the new instance.
        """
        for y in range(self.height):
            for x in range(self.width):
                if isinstance(self.grid[y][x], instance):
                    self.grid[y][x].setting(*args, **kwargs)
    def set_object(self, x, y, sobject_instance):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = sobject_instance

    def get_sobject_count(self, target_instance):
        count = 0
        for co in self.grid:
            for sob in co:
                if isinstance(sob, target_instance):
                    count += 1
        return count

    def print_score(self):
        print(self.score_board)

    def get_object(self, x, y):
        return self.grid[y][x]

    def get_sobjects_by_name(self, sobject_class):
        ins = []
        for i in self.grid:
            for sob in i:
                if isinstance(sob, sobject_class):
                    ins.append(sob)
        return ins

    def get(self):
        return np.array([[self.grid[y][x].id for x in range(self.width)] for y in range(self.height)])

    def draw(self, surface):
        for row in self.grid:
            for sobj in row:
                sobj.draw(surface)
    def reset(self):
        self.grid = [[self.registry.create(0, x, y, self.cell_size) for x in range(self.width)] for y in range(self.height)]
        self.score_board = {
            "reward": 0,
            "score": 0,
        }


class GameEngine(ABC):
    def __init__(self, width, height, cell_size, mode='player'):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.mode = mode
        self.episode_count = 0


        # --- Initialization Order (Template Method) ---
        # 1. Prepare registry
        self.registry = SobjectRegistry()
        self.register_objects(self.registry)  # Hook method

        # 2. Setup screen and field
        window_size = (width * cell_size, height * cell_size)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Grid Game Engine")
        self.field = Field(width, height, cell_size, self.registry)

        # 3. Place objects on the field
        self.player = None
        self.setup_field()  # Hook method

        self.clock = pygame.time.Clock()
        self.running = True

    @abstractmethod
    def register_objects(self, registry: SobjectRegistry):
        """Override this method to register your game's Sobjects."""
        pass

    @abstractmethod
    def setup_field(self):
        """Override this method to place objects on the field."""
        pass

    # --- Core rl_game logic (unchanged) ---
    def set_player(self, player_instance):
        self.player = player_instance
        self.field.set_object(self.player.x, self.player.y, self.player)

    def ai_work(self, model):
        self.model = model
        self.model.learn(10000000)
        self.model.save("model/test")
        print("END!!!")


    def run(self, model=None):
        # (run, handle_events, update, draw methods are the same as before)
        if self.mode == "ai" and model is not None:
            worker_thread = threading.Thread(target=self.ai_work, daemon=True, args=(model,))
            worker_thread.start()  # これ以降、background_workerが裏で動き始める

        while self.running:
            self.handle_events()
            self.draw()
            self.update()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()
        sys.exit()

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