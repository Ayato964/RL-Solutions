import pygame

class Sobject:
    def __init__(self, x, y, id, size, color):
        self.x = x
        self.y = y
        self.id = id
        self.size = size
        self.color = color

    def draw(self, surface):
        rect = pygame.Rect(self.x * self.size, self.y * self.size, self.size, self.size)
        pygame.draw.rect(surface, self.color, rect)