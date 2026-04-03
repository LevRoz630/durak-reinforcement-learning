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
