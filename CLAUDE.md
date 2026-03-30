# Durak Reinforcement Learning

RL agents for the card game Durak. Python game engine with Numba/JIT for speed, Python training pipeline. This is a learning project — the goal is to learn RL, not just produce code.

**Future:** Port engine to C++ (via pybind11) once the Python version is stable and RL training is working.

## Build

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

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

### Engine (learning mode)

The user is learning game engine design. Do not just write complete implementations.

- **Explain** the concept or pattern before writing code
- **Scaffold** the structure: files, classes, function signatures
- **Leave TODOs** for core logic: game rules, state representation, data structures
- When the user writes code, review it and explain what could be improved and why
- Design the engine so it can be ported to C++ later (clean interfaces, minimal Python-specific tricks)

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
  engine/       # Python game engine (Durak rules, game state, card logic)
  training/     # Python training pipeline (agents, environments, training loops)
tests/          # pytest tests
docs/           # research papers, reading notes
```
