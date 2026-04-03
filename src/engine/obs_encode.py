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
            for cid in range(NUM_CARDS):
                if own_hand & (1 << cid):
                    mask[cid] = True
        else:
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
