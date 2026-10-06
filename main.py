"""
Bad Apple Snake Game - Main Entrypoint Delegation.

Allows executing via:
    python -m main
    python -m main eval
    python main.py
    python main.py eval
"""
import sys
from snake_game.main import main

if __name__ == "__main__":
    main(sys.argv)
