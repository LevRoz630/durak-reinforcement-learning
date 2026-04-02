"""Bitmask game state for Numba-accelerated Durak.

State representation (all plain integers / numpy scalars):
    hands[player]   np.int64 bitmask — bit i set = card i in hand
    table_atk       np.int64 bitmask — attack cards currently on table
    table_def       np.int64 bitmask — defense cards currently on table
    discard         np.int64 bitmask — all discarded cards
    trump_suit      np.int8  (0-3)
    phase           np.int8  (0=ATTACK, 1=DEFEND)
    attacker_id     np.int8
    defender_id     np.int8
    num_players     np.int8
    first_defense_completed  np.int8  (0 or 1)
    winner          np.int8  (-1 = no winner yet)

Uncovered attack cards = table_atk & ~table_def
Table ranks: union of suits stripped from table_atk | table_def bits
"""

import numpy as np

from engine.fast_card import (
    NUM_CARDS,
    beats,
    card_rank,
    hand_add,
    hand_contains,
    hand_remove,
    hand_size,
    hand_to_list,
)

HAND_SIZE = 6
PHASE_ATTACK = np.int8(0)
PHASE_DEFEND = np.int8(1)
NO_WINNER = np.int8(-1)


class FastGameState:
    """Pure-integer game state. No Card objects, no lists of tuples."""

    def __init__(self, num_players: int = 2, seed: int | None = None):
        # TODO(human): initialise the state:
        #   - build a shuffled deck as a list of card_ids (0..35)
        #   - set trump_suit from the first card (cards[0])
        #   - deal HAND_SIZE cards to each player into bitmask hands
        #   - determine first attacker (player with lowest trump card, random if none)
        #   - set defender_id, phase, table_atk, table_def, discard, winner
        raise NotImplementedError

    @property
    def uncovered_atk(self) -> int:
        """Bitmask of attack cards not yet covered."""
        # TODO(human): one line using table_atk and table_def
        raise NotImplementedError

    @property
    def table_ranks(self) -> set[int]:
        """Set of ranks (6-14) present on the table (attack + defense side)."""
        # TODO(human): expand table_atk | table_def bits, collect card_rank() for each
        raise NotImplementedError

    @property
    def throw_in_limit(self) -> int:
        """Max attack cards allowed on table this bout."""
        if not self.first_defense_completed:
            return 5
        return hand_size(self.hands[self.defender_id])


class FastGame:
    """Thin action layer over FastGameState (mirrors Game)."""

    def __init__(self, num_players: int = 2, seed: int | None = None):
        self.state = FastGameState(num_players, seed)

    def get_legal_actions(self) -> list:
        """Return legal actions as (action_str, *card_ids).

        TODO(human): port get_legal_actions from Game using bitmask helpers.
        - ATTACK phase, empty table: [("attack", cid), ...]
        - ATTACK phase, cards on table: [("throw_in", cid), ..., ("stop", -1)]
        - DEFEND phase: [("defend", atk_cid, def_cid), ..., ("pick_up", -1)]
        Use hand_to_list() to iterate hands, beats() for defense legality.
        """
        raise NotImplementedError

    def attack(self, cid: int) -> None:
        """Move card from attacker's hand to table_atk."""
        # TODO(human): validate phase, remove from hand, add to table_atk, switch phase
        raise NotImplementedError

    def defend(self, atk_cid: int, def_cid: int) -> None:
        """Cover atk_cid with def_cid.

        TODO(human): validate beats(), remove def_cid from defender hand,
        add to table_def, switch phase to ATTACK.
        """
        raise NotImplementedError

    def throw_in(self, cid: int) -> None:
        """Throw in a card matching a rank already on the table.

        TODO(human): validate rank match (use table_ranks), check throw_in_limit,
        remove from hand, add to table_atk, switch to DEFEND.
        """
        raise NotImplementedError

    def pick_up(self) -> None:
        """Defender picks up all table cards.

        TODO(human): add all table_atk | table_def cards to defender's hand,
        clear table_atk and table_def, draw phase, rotate roles (defender+1 attacks,
        defender+2 defends).
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Attacker ends bout after successful defense.

        TODO(human): guard uncovered_atk != 0, add table cards to discard,
        clear table, set first_defense_completed, draw phase, rotate roles
        (defender becomes attacker, defender+1 becomes defender).
        """
        raise NotImplementedError

    def _draw_phase(self) -> None:
        """Draw up to HAND_SIZE for attacker then defender.

        TODO(human): pop from deck list, hand_add() into bitmask hand,
        stop when deck empty or hand full.
        """
        raise NotImplementedError

    def _check_win(self) -> None:
        """Set winner if any player's hand is empty and deck is empty."""
        # TODO(human): loop hands, check hand_size() == 0 and deck empty
        raise NotImplementedError
