@echo off
title Bad Apple RL Game
cd /d "%~dp0"
echo ========================================================
echo Starting Bad Apple RL Game with Causal Transformer + GRPO
echo ========================================================
".venv\Scripts\python.exe" "snake_game\main.py"
pause
