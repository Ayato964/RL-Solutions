import pygame
from rl_game.engine.game import GameEngine
from rl_game.engine.sobject import Sobject

# --- Game-specific object definitions ---
class Player(Sobject):
    def move(self, dx, dy, field):
        new_x, new_y = self.x + dx, self.y + dy
        if 0 <= new_x < field.width and 0 <= new_y < field.height:
            if field.get_object(new_x, new_y).id != 2:
                void_instance = field.registry.create(0, self.x, self.y, self.size)
                field.set_object(self.x, self.y, void_instance)
                self.x = new_x
                self.y = new_y
                field.set_object(self.x, self.y, self)

    def handle_input(self, event, field):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP: self.move(0, -1, field)
            elif event.key == pygame.K_DOWN: self.move(0, 1, field)
            elif event.key == pygame.K_LEFT: self.move(-1, 0, field)
            elif event.key == pygame.K_RIGHT: self.move(1, 0, field)
            elif event.key == pygame.K_g: print("State:\n", field.get())

class Wall(Sobject):
    pass

# --- Game Definition Class ---
class MyGame(GameEngine):
    def __init__(self):
        # Configure the core rl_game settings
        super().__init__(width=15, height=15, cell_size=20, mode='player')

    def register_objects(self, registry):
        """Register all Sobjects used in this specific game."""
        registry.register(sobject_class=Player, id=1, color=(0, 128, 255))
        registry.register(sobject_class=Wall, id=2, color=(139, 69, 19))
        print("✅ Objects registered.")

    def setup_field(self):
        """Define the initial layout of the game world."""
        # Create and place the player
        player_instance = self.registry.create(id=1, x=1, y=1, size=self.cell_size)
        self.set_player(player_instance)

        # Create and place walls
        wall_positions = [(5, 5), (5, 6), (5, 7), (6, 7)]
        for x, y in wall_positions:
            wall_instance = self.registry.create(id=2, x=x, y=y, size=self.cell_size)
            self.field.set_object(x, y, wall_instance)
        print("✅ Field setup complete.")

# --- Execution Block ---
if __name__ == '__main__':
    # The main block is now clean and simple.
    game = MyGame()
    game.run()