# Durak Reinforcement Learning

RL agents for the card game Durak. C++ game engine with Python training pipeline, bridged via pybind11. This is a learning project — the goal is to learn C++ and RL, not just produce code.

## Build

Single command builds everything (C++ engine via pybind11 + Python package):

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

Python tests only. C++ is tested through the pybind11 bindings.

## Lint & Format

```bash
ruff check src/          # lint
ruff format src/         # format
mypy src/                # type check
```

## Git Workflow

- Feature branches off `main` (e.g. `feature/game-engine`, `feature/training-loop`)
- PRs to merge into `main`

## How Claude Should Help

### C++ (learning mode)

The user is learning C++. Do not just write complete implementations.

- **Explain** the concept or pattern before writing code
- **Scaffold** the structure: files, classes, function signatures, includes
- **Leave TODOs** for core logic: game rules, state representation, data structures
- When the user writes C++, review it and explain what could be improved and why

### RL (learning mode)

The user is learning reinforcement learning. Same approach:

- **Explain** the theory (MDP, policy gradient, NFSP, etc.) before implementing
- **Scaffold** training loops, network architecture boilerplate
- **Leave TODOs** for key decisions: reward shaping, network architecture choices, hyperparameters
- Present trade-offs when multiple valid approaches exist

### General

- Ask before implementing — discuss the approach first
- One plan step at a time, check in before moving on
- When multiple valid approaches exist, present options with trade-offs

## Project Structure

```
src/
  engine/       # C++ game engine (Durak rules, game state, card logic)
  bindings/     # pybind11 bindings exposing engine to Python
  training/     # Python training pipeline (agents, environments, training loops)
tests/          # pytest tests
docs/           # research papers, reading notes
```
