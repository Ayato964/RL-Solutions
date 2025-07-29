import pygame


class SobjectRegistry:
    def __init__(self):
        self._registry = {}
        self._colors = {}
        self.register(Void, 0, (40, 40, 40))

    def register(self, sobject_class, id, color):
        if id in self._registry:
            raise ValueError(f"ID {id} is already registered.")
        self._registry[id] = sobject_class
        self._colors[id] = color

    def create(self, id, x, y, size):
        if id not in self._registry:
            raise ValueError(f"No Sobject registered for ID {id}")

        sobject_class = self._registry[id]
        color = self._colors[id]

        # We pass only the necessary arguments for the base class constructor.
        # Custom Sobjects should handle their own unique attributes.
        return sobject_class(x, y, id, size, color)

    def get_color(self, id):
        return self._colors.get(id)


# A special Void class that the rl_game core needs internally.
class Void:
    def __init__(self, x, y, id, size, color):
        self.x = x
        self.y = y
        self.id = id
        self.size = size
        self.color = color

    def draw(self, surface):
        rect = pygame.Rect(self.x * self.size, self.y * self.size, self.size, self.size)
        pygame.draw.rect(surface, self.color, rect)