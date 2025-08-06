import random

import pygame

from rl_game.engine.game import GameEngine
from rl_game.engine.sobject import Sobject
from stable_baselines3 import DQN
from rl_game.rl.DQN import *
from rl_game.engine.registry import SobjectRegistry, Void
from sobjects import *


class BadAppleTrainer(DQNTrainer):

    def __init__(self, game: GameEngine, action_space, max_id):
        super().__init__(game, action_space, max_id)
        self.max_step = 1000
        self.goal_count = 0

    def reward(self):
        goal_score = 500 if self.game.field.score_board['score'] >= 1000 else 0

        truncated =  self.step_count >= self.max_step
        terminated = self.game.field.score_board['score'] >= 1000

        return -0.1 + goal_score,  truncated, terminated

    def goal(self):
        self.goal_count += 1
        if self.goal_count >= 20:
            self.max_step -= 100
            self.goal_count = 0
            print("-----------------LEVEL UP!!!!--------------------------")


class BadAppleGame(GameEngine):
    def __init__(self, width, height, cell_size):
        super().__init__(width, height, cell_size, mode="ai")
        self.font = pygame.font.SysFont("", 48)

    def register_objects(self, registry:SobjectRegistry):
        registry.registers(Register())


    def setup_field(self):
        player = self.registry.create(id=1, x=5, y=3, size=35)
        self.set_player(player)
        top = [(0, i) for i in range(self.field.width)]
        down = [(self.field.height-1, i) for i in range(self.field.width)]
        top += down
        for y, x in top:
            wall = self.registry.create(id=2, x=x, y=y, size=35)
            self.field.set_object(x, y, wall)

        wall_c = (0, self.field.width-1)
        for y in range(1, self.field.height):
            wall_left = self.registry.create(id=2, x=wall_c[0], y=y, size=35)
            wall_right = self.registry.create(id=2, x=wall_c[1], y=y, size=35)
            self.field.set_object(x=wall_c[0], y=y, sobject_instance=wall_left)
            self.field.set_object(x=wall_c[1], y=y, sobject_instance=wall_right)

        wall2 = [(10, 5), (10, 6),(10, 7), (10, 8),(10, 9), (10, 10),(10, 11), (10, 12),
                 (20, 5), (20, 6),(20, 7), (20, 8),(20, 9), (20, 10),(20, 11), (20, 12),]
        for x, y in wall2:
            wall_instance = self.registry.create(id=2, x=x, y=y, size=self.cell_size)
            self.field.set_object(x, y, wall_instance)

        self.field.score_board['apple_count'] = 0
        self.field.score_board['apple_max'] = 10
        self.add_apple()

    def update(self):
        if self.field.score_board['apple_count'] != self.field.score_board['apple_max']:
            self.add_apple()
        self.draw_text(f"SCORE:{self.field.score_board['score']} / 1000  Episode:{self.episode_count}", self.font, 20, 25)



    def add_apple(self):
        x = random.randint(0, self.width-1)
        y = random.randint(0, self.height-1)
        if isinstance(self.field.get_object(x, y), Void):
            rew = self.registry.create(3, x, y, size=35)
            self.field.set_object(x, y, rew)
            self.field.score_board['apple_count'] += 1


if __name__ == "__main__":
    game = BadAppleGame(30, 20, 35)

    trainer = BadAppleTrainer(game, 4, 4)
    policy_kwargs = dict(
        features_extractor_class=AIPlayer,
        features_extractor_kwargs=dict(features_dim=4),  # CustomCNNの__init__に渡す引数
    )
    model = DQN(
        "MlpPolicy",
        trainer,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./dqn_custom_log/"
    )
    model.load("model/test")

    game.run(model)