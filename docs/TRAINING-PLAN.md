# DQN Training Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up minimum infrastructure to train one DQN agent against a random opponent in 2-player Durak.

**Architecture:** Python training loop with Numba-accelerated inner functions (obs encoding, legal mask) and a PyTorch Q-network. FastGame handles game state; a thin env layer translates between game concepts and RL concepts (obs vectors, integer action indices, rewards).

**Tech Stack:** Python 3.12, Numba, NumPy, PyTorch

**Spec:** `docs/superpowers/specs/2026-04-03-dqn-training-infra-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add `numba` to dependencies |
| `src/engine/fast_card.py` | Modify | Add `@njit` to all 9 primitive functions |
| `src/engine/obs_encode.py` | Create | `encode_observation()`, `legal_action_mask()` — both `@njit` |
| `src/training/__init__.py` | Create | Empty package marker |
| `src/training/env.py` | Create | `is_agent_turn()`, `apply_action()`, `get_obs()`, `get_mask()` — Python wrappers over @njit functions |
| `src/training/random_opponent.py` | Create | `random_action(mask)` — uniform sample from legal actions |
| `src/training/replay_buffer.py` | Create | Circular numpy replay buffer |
| `src/training/dqn_agent.py` | Create | `QNetwork` (PyTorch), `DQNAgent` (select + update + sync) |
| `src/training/train.py` | Create | Full training loop |
| `src/test_obs_encode.py` | Create | Tests for encode_observation and legal_action_mask |
| `src/test_replay_buffer.py` | Create | Tests for ReplayBuffer |
| `src/test_dqn_agent.py` | Create | Tests for QNetwork forward pass and action selection |
| `src/test_train.py` | Create | Smoke test: N steps without crash |

---

## Chunk 1: Numba Backend

### Task 1: Add numba to dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add numba to pyproject.toml dependencies**

In `pyproject.toml`, add `"numba>=0.60"` to the `dependencies` list.

- [ ] **Step 2: Install**

```bash
pip install numba
```

Expected: numba installs successfully.

- [ ] **Step 3: Verify**

```bash
python3 -c "import numba; print(numba.__version__)"
```

Expected: prints a version string.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add numba dependency"
```

---

### Task 2: Apply @njit to fast_card.py

**Files:**
- Modify: `src/engine/fast_card.py`

- [ ] **Step 1: Run existing tests to establish baseline**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_fast_game.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Add @njit decorators**

Add `from numba import njit` at the top of `src/engine/fast_card.py`.

Decorate all 9 functions with `@njit`:
`card_id`, `card_suit`, `card_rank`, `beats`, `hand_contains`, `hand_add`, `hand_remove`, `hand_size`, `hand_to_list`.

Note: `hand_to_list` uses `np.empty` which Numba supports. No changes to function bodies needed.

- [ ] **Step 3: Run tests again**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_fast_game.py -v
```

Expected: all tests still pass. First run will be slow (JIT compilation). Subsequent runs are fast.

- [ ] **Step 4: Commit**

```bash
git add src/engine/fast_card.py pyproject.toml
git commit -m "feat: add @njit to fast_card primitives, add numba dependency"
```

---

### Task 3: obs_encode.py — encode_observation

**Files:**
- Create: `src/engine/obs_encode.py`
- Create: `src/test_obs_encode.py`

- [ ] **Step 1: Write the failing test**

Create `src/test_obs_encode.py`:

```python
import unittest
import numpy as np
from engine.fast_game_state import FastGame, PHASE_ATTACK, PHASE_DEFEND
from engine.fast_card import hand_size


class TestEncodeObservation(unittest.TestCase):
    def test_shape(self):
        from engine.obs_encode import encode_observation
        from engine.fast_game_state import FastGameState
        s = FastGameState(num_players=2, seed=7)
        obs = encode_observation(
            int(s.hands[0]), int(s.table_atk), int(s.table_def),
            int(s.discard), int(s.trump_suit),
            hand_size(int(s.hands[1])), int(s.phase),
        )
        self.assertEqual(obs.shape, (150,))
        self.assertEqual(obs.dtype, np.float32)

    def test_own_hand_bits(self):
        from engine.obs_encode import encode_observation
        from engine.fast_game_state import FastGameState
        from engine.fast_card import hand_to_list
        s = FastGameState(num_players=2, seed=7)
        obs = encode_observation(
            int(s.hands[0]), int(s.table_atk), int(s.table_def),
            int(s.discard), int(s.trump_suit),
            hand_size(int(s.hands[1])), int(s.phase),
        )
        for cid in hand_to_list(int(s.hands[0])):
            self.assertEqual(obs[cid], 1.0, f"card {cid} should be in hand")

    def test_empty_table_zero(self):
        from engine.obs_encode import encode_observation
        from engine.fast_game_state import FastGameState
        s = FastGameState(num_players=2, seed=7)
        obs = encode_observation(
            int(s.hands[0]), int(s.table_atk), int(s.table_def),
            int(s.discard), int(s.trump_suit),
            hand_size(int(s.hands[1])), int(s.phase),
        )
        # table_atk slice should be all zeros at game start
        self.assertTrue(np.all(obs[36:72] == 0.0))

    def test_trump_suit_one_hot(self):
        from engine.obs_encode import encode_observation
        from engine.fast_game_state import FastGameState
        s = FastGameState(num_players=2, seed=7)
        obs = encode_observation(
            int(s.hands[0]), int(s.table_atk), int(s.table_def),
            int(s.discard), int(s.trump_suit),
            hand_size(int(s.hands[1])), int(s.phase),
        )
        trump_slice = obs[144:148]
        self.assertEqual(trump_slice.sum(), 1.0)
        self.assertEqual(trump_slice[int(s.trump_suit)], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_obs_encode.py -v
```

Expected: `ModuleNotFoundError: No module named 'engine.obs_encode'`

- [ ] **Step 3: Implement encode_observation**

Create `src/engine/obs_encode.py`:

```python
"""Observation encoding and legal action masking for DQN agent.

Observation vector layout (150 float32):
    [0:36]   own hand          — one-hot over 36 card IDs
    [36:72]  table attack      — one-hot over 36 card IDs
    [72:108] table defense     — one-hot over 36 card IDs
    [108:144] discard pile     — one-hot over 36 card IDs
    [144:148] trump suit       — one-hot over 4 suits
    [148]    opponent hand size — float, normalised by HAND_SIZE (6)
    [149]    phase             — 0.0 = ATTACK, 1.0 = DEFEND

Action space (38 integers):
    0–35   play card with that card_id (attack / throw_in / defend — context resolves which)
    36     stop
    37     pick_up
"""

import numpy as np
from numba import njit

from engine.fast_card import NUM_CARDS, beats, card_rank

OBS_DIM = 150
ACTION_DIM = 38
STOP = 36
PICK_UP = 37
HAND_SIZE = 6


@njit
def encode_observation(
    own_hand: int,
    table_atk: int,
    table_def: int,
    discard: int,
    trump_suit: int,
    opp_hand_size: int,
    phase: int,
) -> np.ndarray:
    """Encode FastGameState fields into a float32 observation vector."""
    obs = np.zeros(150, dtype=np.float32)
    for cid in range(NUM_CARDS):
        if own_hand & (1 << cid):
            obs[cid] = 1.0
        if table_atk & (1 << cid):
            obs[36 + cid] = 1.0
        if table_def & (1 << cid):
            obs[72 + cid] = 1.0
        if discard & (1 << cid):
            obs[108 + cid] = 1.0
    obs[144 + trump_suit] = 1.0
    obs[148] = opp_hand_size / 6.0
    obs[149] = float(phase)
    return obs


@njit
def legal_action_mask(
    own_hand: int,
    table_atk: int,
    table_def_for: np.ndarray,
    trump_suit: int,
    phase: int,
    table_ranks: int,
    num_table_atk: int,
    throw_in_limit: int,
) -> np.ndarray:
    """Return bool[38] mask: True where action index is legal."""
    mask = np.zeros(38, dtype=np.bool_)

    if phase == 0:  # ATTACK
        if num_table_atk == 0:
            # Empty table: must attack with a card from hand
            for cid in range(NUM_CARDS):
                if own_hand & (1 << cid):
                    mask[cid] = True
        else:
            # Can throw in matching ranks or stop
            for cid in range(NUM_CARDS):
                if own_hand & (1 << cid):
                    if table_ranks & (1 << (card_rank(cid) - 6)):
                        if num_table_atk < throw_in_limit:
                            mask[cid] = True
            mask[36] = True  # stop always legal when table non-empty
    else:  # DEFEND
        for atk_cid in range(NUM_CARDS):
            if (table_atk & (1 << atk_cid)) and table_def_for[atk_cid] == -1:
                for def_cid in range(NUM_CARDS):
                    if own_hand & (1 << def_cid):
                        if beats(def_cid, atk_cid, trump_suit):
                            mask[def_cid] = True
        mask[37] = True  # pick_up always legal
    return mask
```

- [ ] **Step 4: Run tests**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_obs_encode.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Add legal_action_mask tests**

Add to `src/test_obs_encode.py`:

```python
class TestLegalActionMask(unittest.TestCase):
    def test_attack_phase_empty_table(self):
        from engine.obs_encode import legal_action_mask
        from engine.fast_game_state import FastGameState
        from engine.fast_card import hand_to_list, hand_size
        s = FastGameState(num_players=2, seed=7)
        # Game starts in ATTACK phase with empty table
        mask = legal_action_mask(
            int(s.hands[s.attacker_id]), int(s.table_atk),
            s.table_def_for, int(s.trump_suit), int(s.phase),
            s.table_ranks, hand_size(int(s.table_atk)), s.throw_in_limit,
        )
        self.assertEqual(mask.shape, (38,))
        # All hand cards should be legal
        for cid in hand_to_list(int(s.hands[s.attacker_id])):
            self.assertTrue(mask[cid], f"card {cid} should be legal to attack")
        # Stop and pick_up should not be legal on empty table
        self.assertFalse(mask[36])  # stop
        self.assertFalse(mask[37])  # pick_up

    def test_defend_phase_pick_up_always_legal(self):
        from engine.obs_encode import legal_action_mask
        from engine.fast_game_state import FastGame, PHASE_DEFEND
        from engine.fast_card import hand_size
        game = FastGame(num_players=2, seed=7)
        # Force into defend phase by attacking
        actions = game.get_legal_actions()
        atk = next(a for a in actions if a[0] == "attack")
        game.attack(atk[1])
        s = game.state
        mask = legal_action_mask(
            int(s.hands[s.defender_id]), int(s.table_atk),
            s.table_def_for, int(s.trump_suit), int(s.phase),
            s.table_ranks, hand_size(int(s.table_atk)), s.throw_in_limit,
        )
        self.assertTrue(mask[37])  # pick_up always legal in DEFEND
        self.assertFalse(mask[36])  # stop not legal in DEFEND
```

- [ ] **Step 6: Run all obs_encode tests**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_obs_encode.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/engine/obs_encode.py src/test_obs_encode.py
git commit -m "feat: add obs_encode with @njit encode_observation and legal_action_mask"
```

---

### Task 4: env.py — Python wrappers and action application

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/env.py`

- [ ] **Step 1: Create training package**

Create empty `src/training/__init__.py`.

- [ ] **Step 2: Implement env.py**

Create `src/training/env.py`:

```python
"""Thin RL environment layer over FastGame.

Translates between game engine concepts (attacker/defender, bitmasks)
and RL concepts (obs vectors, integer action indices, rewards).

Action index encoding:
    0–35  play card with that card_id
    36    stop
    37    pick_up
"""

import numpy as np

from engine.fast_card import beats, hand_size, hand_to_list
from engine.fast_game_state import PHASE_ATTACK, PHASE_DEFEND, FastGame, FastGameState
from engine.obs_encode import OBS_DIM, encode_observation, legal_action_mask


def current_player(state: FastGameState) -> int:
    """Return the player_id whose turn it is."""
    if state.phase == PHASE_ATTACK:
        return int(state.attacker_id)
    return int(state.defender_id)


def get_obs(state: FastGameState, agent_id: int) -> np.ndarray:
    """Encode game state as float32[150] observation for agent_id."""
    opp_id = 1 - agent_id
    return encode_observation(
        int(state.hands[agent_id]),
        int(state.table_atk),
        int(state.table_def),
        int(state.discard),
        int(state.trump_suit),
        hand_size(int(state.hands[opp_id])),
        int(state.phase),
    )


def get_mask(state: FastGameState) -> np.ndarray:
    """Return legal action mask bool[38] for the current player."""
    player = current_player(state)
    return legal_action_mask(
        int(state.hands[player]),
        int(state.table_atk),
        state.table_def_for,
        int(state.trump_suit),
        int(state.phase),
        state.table_ranks,
        hand_size(int(state.table_atk)),
        state.throw_in_limit,
    )


def apply_action(game: FastGame, action_idx: int) -> None:
    """Apply a flat action index to the game.

    0–35: play card cid
        ATTACK + empty table  → attack(cid)
        ATTACK + cards on table → throw_in(cid)
        DEFEND → defend(first uncovered atk_cid that cid beats, cid)
    36: stop()
    37: pick_up()
    """
    s = game.state
    if action_idx == 37:
        game.pick_up()
    elif action_idx == 36:
        game.stop()
    else:
        cid = action_idx
        if s.phase == PHASE_ATTACK:
            if not s.table_atk:
                game.attack(cid)
            else:
                game.throw_in(cid)
        else:
            for atk_cid in hand_to_list(s.uncovered_atk):
                if beats(cid, int(atk_cid), int(s.trump_suit)):
                    game.defend(int(atk_cid), cid)
                    return
```

- [ ] **Step 3: Verify imports work**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -c "from training.env import get_obs, get_mask, apply_action; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/training/__init__.py src/training/env.py
git commit -m "feat: add training env wrapper (get_obs, get_mask, apply_action)"
```

---

## Chunk 2: Training Infrastructure

### Task 5: random_opponent.py

**Files:**
- Create: `src/training/random_opponent.py`

- [ ] **Step 1: Implement**

Create `src/training/random_opponent.py`:

```python
import numpy as np


def random_action(mask: np.ndarray) -> int:
    """Return a uniformly random legal action index."""
    legal = np.where(mask)[0]
    return int(np.random.choice(legal))
```

- [ ] **Step 2: Quick sanity check**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -c "
import numpy as np
from training.random_opponent import random_action
mask = np.zeros(38, dtype=bool)
mask[5] = mask[12] = mask[36] = True
for _ in range(20):
    a = random_action(mask)
    assert a in (5, 12, 36), f'illegal action {a}'
print('ok')
"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/training/random_opponent.py
git commit -m "feat: add random_opponent"
```

---

### Task 6: replay_buffer.py

**Files:**
- Create: `src/training/replay_buffer.py`
- Create: `src/test_replay_buffer.py`

- [ ] **Step 1: Write failing tests**

Create `src/test_replay_buffer.py`:

```python
import unittest
import numpy as np


class TestReplayBuffer(unittest.TestCase):
    def _make(self, capacity=100):
        from training.replay_buffer import ReplayBuffer
        return ReplayBuffer(capacity=capacity, obs_dim=150)

    def test_push_and_len(self):
        buf = self._make()
        obs = np.zeros(150, dtype=np.float32)
        buf.push(obs, 5, 1.0, obs, False)
        self.assertEqual(len(buf), 1)

    def test_circular_wrap(self):
        buf = self._make(capacity=10)
        obs = np.zeros(150, dtype=np.float32)
        for i in range(15):
            buf.push(obs, i % 38, 0.0, obs, False)
        self.assertEqual(len(buf), 10)  # capped at capacity

    def test_sample_shape(self):
        buf = self._make()
        obs = np.random.rand(150).astype(np.float32)
        for _ in range(20):
            buf.push(obs, 3, 0.0, obs, False)
        batch = buf.sample(8)
        obs_b, act_b, rew_b, nobs_b, done_b = batch
        self.assertEqual(obs_b.shape, (8, 150))
        self.assertEqual(act_b.shape, (8,))
        self.assertEqual(rew_b.shape, (8,))

    def test_sample_requires_enough_data(self):
        buf = self._make()
        obs = np.zeros(150, dtype=np.float32)
        buf.push(obs, 0, 0.0, obs, False)
        with self.assertRaises(ValueError):
            buf.sample(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_replay_buffer.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/training/replay_buffer.py`:

```python
import numpy as np


class ReplayBuffer:
    """Circular replay buffer storing (obs, action, reward, next_obs, done) tuples."""

    def __init__(self, capacity: int, obs_dim: int = 150):
        self.capacity = capacity
        self._pos = 0
        self._size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        self.obs[self._pos] = obs
        self.next_obs[self._pos] = next_obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        if batch_size > self._size:
            raise ValueError(f"Not enough data: have {self._size}, need {batch_size}")
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self._size
```

- [ ] **Step 4: Run tests**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_replay_buffer.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/replay_buffer.py src/test_replay_buffer.py
git commit -m "feat: add circular replay buffer"
```

---

### Task 7: dqn_agent.py

**Files:**
- Create: `src/training/dqn_agent.py`
- Create: `src/test_dqn_agent.py`

- [ ] **Step 1: Write failing tests**

Create `src/test_dqn_agent.py`:

```python
import unittest
import numpy as np


class TestQNetwork(unittest.TestCase):
    def test_output_shape(self):
        import torch
        from training.dqn_agent import QNetwork
        net = QNetwork(obs_dim=150, action_dim=38)
        x = torch.zeros(4, 150)  # batch of 4
        out = net(x)
        self.assertEqual(out.shape, (4, 38))


class TestDQNAgent(unittest.TestCase):
    def _make(self):
        from training.dqn_agent import DQNAgent
        return DQNAgent(obs_dim=150, action_dim=38)

    def test_select_action_respects_mask(self):
        agent = self._make()
        obs = np.zeros(150, dtype=np.float32)
        mask = np.zeros(38, dtype=bool)
        mask[7] = True  # only action 7 is legal
        action = agent.select_action(obs, mask, epsilon=0.0)
        self.assertEqual(action, 7)

    def test_select_action_random_is_legal(self):
        agent = self._make()
        obs = np.zeros(150, dtype=np.float32)
        mask = np.zeros(38, dtype=bool)
        mask[3] = mask[15] = mask[36] = True
        for _ in range(50):
            action = agent.select_action(obs, mask, epsilon=1.0)
            self.assertIn(action, [3, 15, 36])

    def test_update_returns_loss(self):
        import numpy as np
        agent = self._make()
        batch = (
            np.zeros((8, 150), dtype=np.float32),
            np.zeros(8, dtype=np.int64),
            np.zeros(8, dtype=np.float32),
            np.zeros((8, 150), dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        )
        loss = agent.update(batch)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_dqn_agent.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/training/dqn_agent.py`:

```python
import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    # TODO(human): choose your architecture — hidden_dim, num_layers, activation.
    # Starting point: 2 hidden layers of 256 with ReLU.
    # Wider networks learn more complex strategies but train slower.
    def __init__(self, obs_dim: int = 150, action_dim: int = 38, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int = 150,
        action_dim: int = 38,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        # TODO(human): tune lr, gamma. lr=1e-3 is aggressive; 1e-4 is safer.
        self.gamma = gamma
        self.device = device

        self.q_online = QNetwork(obs_dim, action_dim).to(device)
        self.q_target = QNetwork(obs_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection with illegal action masking."""
        if np.random.random() < epsilon:
            return int(np.random.choice(np.where(mask)[0]))
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_online(obs_t).squeeze(0).cpu().numpy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def update(self, batch: tuple) -> float:
        """One DQN gradient step. Returns scalar loss."""
        obs, actions, rewards, next_obs, dones = batch
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.LongTensor(actions).to(self.device)
        rew_t = torch.FloatTensor(rewards).to(self.device)
        nobs_t = torch.FloatTensor(next_obs).to(self.device)
        done_t = torch.FloatTensor(dones).to(self.device)

        q_vals = self.q_online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(nobs_t).max(1)[0]
            targets = rew_t + self.gamma * next_q * (1 - done_t)

        loss = nn.functional.mse_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def sync_target(self) -> None:
        """Copy online weights to target network."""
        self.q_target.load_state_dict(self.q_online.state_dict())
```

- [ ] **Step 4: Run tests**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_dqn_agent.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/dqn_agent.py src/test_dqn_agent.py
git commit -m "feat: add QNetwork and DQNAgent"
```

---

### Task 8: train.py — training loop

**Files:**
- Create: `src/training/train.py`
- Create: `src/test_train.py`

- [ ] **Step 1: Write smoke test**

Create `src/test_train.py`:

```python
import unittest


class TestTrainSmoke(unittest.TestCase):
    def test_runs_without_crash(self):
        """Training loop runs for a small number of steps without error."""
        from training.train import train
        stats = train(
            num_episodes=5,
            buffer_capacity=500,
            batch_size=16,
            learning_starts=32,
            target_update_freq=50,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.99,
        )
        self.assertIn("episodes", stats)
        self.assertEqual(stats["episodes"], 5)

    def test_steps_accumulate(self):
        """Training loop accumulates steps across episodes (each game takes >0 steps)."""
        from training.train import train
        stats = train(num_episodes=3, learning_starts=1000)
        self.assertGreater(stats["steps"], 3)  # at minimum 1 step per episode


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_train.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/training/train.py`:

```python
"""DQN training loop: one agent (player 0) vs random opponent (player 1)."""

from engine.fast_game_state import FastGame
from training.dqn_agent import DQNAgent
from training.env import apply_action, current_player, get_mask, get_obs
from training.random_opponent import random_action
from training.replay_buffer import ReplayBuffer

AGENT_ID = 0


def train(
    num_episodes: int = 10_000,
    buffer_capacity: int = 100_000,
    batch_size: int = 64,
    learning_starts: int = 1_000,
    target_update_freq: int = 500,
    # TODO(human): tune these — epsilon controls exploration vs exploitation.
    # High start = explore broadly early; fast decay = commit to learned policy sooner.
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
) -> dict:
    agent = DQNAgent()
    buf = ReplayBuffer(capacity=buffer_capacity)
    epsilon = epsilon_start
    step = 0

    for episode in range(num_episodes):
        game = FastGame(num_players=2)

        # If opponent goes first, auto-step them until it's the agent's turn
        while current_player(game.state) != AGENT_ID and game.state.winner == -1:
            opp_mask = get_mask(game.state)
            apply_action(game, random_action(opp_mask))

        obs = get_obs(game.state, AGENT_ID)
        mask = get_mask(game.state)

        while game.state.winner == -1:
            action = agent.select_action(obs, mask, epsilon)
            apply_action(game, action)

            # Opponent plays until it's the agent's turn again (or game ends)
            while current_player(game.state) != AGENT_ID and game.state.winner == -1:
                opp_mask = get_mask(game.state)
                apply_action(game, random_action(opp_mask))

            next_obs = get_obs(game.state, AGENT_ID)
            next_mask = get_mask(game.state)
            done = game.state.winner != -1
            reward = 1.0 if game.state.winner == AGENT_ID else (-1.0 if done else 0.0)

            buf.push(obs, action, reward, next_obs, done)
            obs, mask = next_obs, next_mask
            step += 1

            if len(buf) >= learning_starts:
                agent.update(buf.sample(batch_size))

            if step % target_update_freq == 0:
                agent.sync_target()

        # Decay once per episode (not per step) — keeps exploration high during
        # early episodes when the buffer is sparse.
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return {"episodes": num_episodes, "steps": step}
```

- [ ] **Step 4: Run smoke tests**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest test_train.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/vscode/repos/durak-reinforcement-learning/src && python3 -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/training/train.py src/test_train.py
git commit -m "feat: add DQN training loop"
```
