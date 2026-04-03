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
