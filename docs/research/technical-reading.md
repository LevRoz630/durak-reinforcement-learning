# Technical Reading Plan

One-day (~6.5h) reading plan covering RL theory relevant to building a Durak agent with self-play and imperfect information.

## 1. RL Foundations (2.5h)

### Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed)
Free: incompleteideas.net/book/the-book-2nd.html

- **Ch 3.1–3.6** — MDP formalism, Bellman equation, returns
- **Ch 6.1–6.5** — TD learning, SARSA, Q-learning
- **Ch 9.1–9.4** — Function approximation (why NNs replace tables)
- **Ch 13.1–13.4** — Policy gradient theorem, REINFORCE

### DQN
- **Mnih et al. 2013** "Playing Atari with Deep RL" — sections 1-4. Experience replay, target networks.
- **Mnih et al. 2015** (Nature) — Methods section only for the clearest algorithm description.

## 2. Self-Play (1.5h)

- **Silver et al. 2016 (AlphaGo)** — sections 1-3. Policy + value network combination.
- **Silver et al. 2018 (AlphaZero)** — full paper. Self-play training loop: play -> collect (state, policy, outcome) -> train -> repeat. Assumes perfect info.
- **Jonathan Hui, "Self-Play in RL"** (Medium) — Nash equilibria connection, strategy cycling pitfalls.

## 3. Imperfect Information (2h)

Core block for Durak — hidden cards make this fundamentally different from Chess/Go.

### CFR
- **Zinkevich et al. 2007** "Regret Minimization in Games with Incomplete Information" — sections 1-3. Information sets: states indistinguishable due to hidden info. CFR operates over these.

### Poker AI
- **Brown & Sandholm 2018 (Libratus)** — main body. Game abstraction (card/bet abstraction) for tractability.
- **Brown & Sandholm 2019 (Pluribus)** — main body. Why MCTS fails with hidden info; depth-limited search with learned values.

### Multi-agent theory
- **Lanctot et al. 2017** "A Unified Game-Theoretic Approach to Multiagent RL" — sections 1-3, 5. Why naive self-play can fail; PSRO framework.

## 4. Algorithm Selection (40min)

- **Lanctot et al. 2019 (OpenSpiel)** — sections 1-3, Table 1. Maps game properties to applicable algorithms.
- **Heinrich & Silver 2016 (NFSP)** — sections 1-4. Most relevant to Durak: combines DQN with fictitious play for imperfect-info games. Two networks (best-response + average policy), converges toward Nash equilibrium.

## Candidate Approaches for Durak

1. **NFSP** — best fit for card games, proven on poker
2. **Deep CFR** — neural net CFR (bonus: Brown et al. 2019)
3. **AlphaZero-style + info set abstraction** — simpler but less principled for hidden info
