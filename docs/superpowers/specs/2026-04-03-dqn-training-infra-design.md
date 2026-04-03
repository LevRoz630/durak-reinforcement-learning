# DQN Training Infrastructure Design

**Date:** 2026-04-03
**Scope:** Get one DQN agent working against a random opponent in Durak (2-player).

---

## Goal

Stand up the minimum infrastructure needed to train a single DQN agent against a random
opponent. The emphasis is on correctness and learnability first — speed optimisations
are secondary at this stage.

---

## Architecture

```
Python training loop
  ├── FastGame (game engine)         ← bitmask state, already implemented
  ├── Numba @njit functions          ← fast_card.py primitives + new helpers
  ├── Observation encoder            ← bitmask state → float[151] vector
  ├── Action mask                    ← bool[38], True where action is legal
  ├── DQN agent (PyTorch)            ← Q-network + replay buffer + target net
  └── Random opponent                ← uniform sample from legal actions
```

---

## Numba Backend

### What gets `@njit`

All functions in `fast_card.py` are pure integer arithmetic and can be decorated
immediately with no refactoring:

- `card_id`, `card_suit`, `card_rank`
- `beats`
- `hand_contains`, `hand_add`, `hand_remove`, `hand_size`, `hand_to_list`

New functions to write and `@njit`:

- `encode_observation(state_arrays...) -> float32[151]` — hot path, called every step
- `legal_action_mask(state_arrays...) -> bool[38]` — called every step

State transition methods (`attack`, `defend`, `throw_in`, `stop`, `pick_up`) in
`FastGame` are also called every step. These stay as Python methods for now — the
class structure makes `@jitclass` impractical and the per-call overhead is small
compared to the NN forward pass.

### What does NOT get `@njit`

| Component | Reason |
|---|---|
| `FastGameState.__init__` | Uses Python `random` and `list`; called once per game |
| `FastGame._draw_phase` | Deck is a Python `list[int]` |
| Training loop | Contains PyTorch NN calls |
| Replay buffer | Python/NumPy — not a bottleneck |

---

## Action Space

Flat integer space, size 38:

```
0–35   play card with card_id N   (attack / throw_in / defend — context determines which)
36     stop
37     pick_up
```

Legal actions are communicated as a `bool[38]` mask. For defense, the agent picks a
card from hand; the game resolves which uncovered attack card it covers (first valid
match). This loses the ability to express "cover attack card X specifically with card Y"
but is sufficient for a first agent and keeps the action space simple.

The DQN network outputs 38 Q-values. Before action selection, illegal action Q-values
are set to `-inf` so argmax always returns a legal action.

---

## Observation Vector

The agent has imperfect information — it does not see the opponent's hand.

| Feature | Encoding | Size |
|---|---|---|
| Own hand | one-hot over 36 card IDs | 36 |
| Table attack cards | one-hot over 36 card IDs | 36 |
| Table defense cards | one-hot over 36 card IDs | 36 |
| Discard pile | one-hot over 36 card IDs | 36 |
| Trump suit | one-hot over 4 suits | 4 |
| Opponent hand size | scalar (normalised 0–1 by max hand size) | 1 |
| Phase (attack/defend) | scalar (0 or 1) | 1 |

**Total: 150 floats** (`float32`)

All encoded from bitmasks already present in `FastGameState` — no extra state needed.

---

## DQN Training Loop

```
initialise Q-network (online) and Q-target (frozen copy)
initialise replay buffer (capacity C)
epsilon = epsilon_start

for episode in range(num_episodes):
    game = FastGame(num_players=2)
    obs = encode_observation(game.state)

    while game.state.winner == -1:
        mask = legal_action_mask(game.state)

        # Agent acts (epsilon-greedy)
        if random() < epsilon:
            action = random choice from legal actions
        else:
            q_values = Q_online(obs)
            q_values[~mask] = -inf
            action = argmax(q_values)

        # Random opponent acts when it's their turn (handled inside step)
        apply_action(game, action)

        next_obs = encode_observation(game.state)
        reward = terminal_reward(game.state)   # +1 win, -1 lose, 0 otherwise

        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs

        # Train
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = dqn_loss(Q_online, Q_target, batch)
            optimiser.step(loss)

        # Sync target network every K steps
        if step % target_update_freq == 0:
            Q_target.load_state_dict(Q_online.state_dict())

    epsilon = max(epsilon_end, epsilon * epsilon_decay)
```

---

## Reward

Sparse terminal reward only:

```
+1   agent wins (opponent's hand empties first)
-1   agent loses
 0   all other steps
 
```

**Known issue:** sparse reward makes credit assignment hard — the agent gets no signal
until the game ends (typically 20–80 steps). Reward shaping is a future option but is
intentionally excluded here to keep the first implementation simple.

---

## Known Disadvantages

**1. Agent trained against random overfits to random play.**
A DQN trained solely against a random opponent learns to exploit random mistakes, not
to play principled Durak. It may perform poorly against any non-random opponent. This
is acceptable as a first milestone — self-play comes later.

**2. Sparse reward slows convergence.**
No intermediate signal means the Q-network receives very infrequent gradient signal
from actual game outcomes. Expect slow convergence and noisy early training.

**3. Defense expressiveness loss.**
The agent cannot express "cover attack card X with defense card Y" — it picks a defense
card and the system resolves the pairing. This is fine while only one attack card is
uncovered at a time, but may produce suboptimal play in multi-card bouts.

**4. `@njit` on state transitions deferred.**
`FastGame` methods stay as Python class methods. Crossing the Python/Numba boundary
on every step adds overhead. Acceptable for a first agent; revisit if rollout speed
becomes the bottleneck.

**5. No opponent modelling.**
The observation vector encodes opponent hand *size* but not opponent hand *contents*
(hidden information). The agent cannot reason about what the opponent might hold.
This is correct behaviour for an agent with imperfect information, but limits the
depth of strategy the agent can learn.

---

## Files to Create

```
src/
  engine/
    fast_card.py          ← add @njit decorators (existing file)
    obs_encode.py         ← encode_observation(), legal_action_mask()
  training/
    __init__.py
    replay_buffer.py      ← circular replay buffer
    dqn_agent.py          ← Q-network, epsilon-greedy, loss
    train.py              ← training loop
    random_opponent.py    ← uniform random legal action selection
```

---

## Success Criterion

The agent wins against the random opponent more than 60% of games after training,
measured over 1000 evaluation episodes with epsilon = 0.
