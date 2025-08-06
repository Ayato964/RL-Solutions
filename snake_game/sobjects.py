from typing import List

import pygame

from rl_game.engine.registry import RegistryList, RegistryObject
from rl_game.engine.sobject import Sobject
from rl_game.engine.handler import *

class Player(Sobject, PlayerHandler):

    def move(self, dx, dy, field):
        new_x, new_y = self.x + dx, self.y + dy

        if isinstance(field.grid[new_y][new_x], ItemHandler):
            field.grid[new_y][new_x].pickup(self, field)

        if 0 <= new_x < field.width and 0 <= new_y < field.height:
            if field.get_object(new_x, new_y).id != 2:
                void_instance = field.registry.create(0, self.x, self.y, self.size)
                field.set_object(self.x, self.y, void_instance)
                self.x = new_x
                self.y = new_y
                field.set_object(self.x, self.y, self)
            else:
                field.score_board['reward'] -= 10

    def handle_input(self, event, field):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP: self.move(0, -1, field)
            elif event.key == pygame.K_DOWN: self.move(0, 1, field)
            elif event.key == pygame.K_LEFT: self.move(-1, 0, field)
            elif event.key == pygame.K_RIGHT: self.move(1, 0, field)
            elif event.key == pygame.K_g: print("State:\n", field.get())


    def handle_AI_input(self, event, field):
        if event == 0:
            self.move(0, -1, field)
        elif event == 1:
            self.move(0, 1, field)
        elif event == 2:
            self.move(-1, 0, field)
        elif event == 3:
            self.move(1, 0, field)


class Wall(Sobject):
    pass


class RewardApple(Sobject, ItemHandler):
    def pickup(self, target, field):
        field.score_board['reward'] += 20
        field.score_board['score'] += 100
        field.score_board['apple_count'] -= 1
        #field.print_score()

class Register(RegistryList):
    def init(self, registry: List[RegistryObject]):
        registry.append(RegistryObject(sobject_class=Player, id=1, color=(0, 125, 0)))
        registry.append(RegistryObject(sobject_class=Wall, id=2, color=(0, 0, 255)))
        registry.append(RegistryObject(RewardApple, 3, (255, 0, 0)))
