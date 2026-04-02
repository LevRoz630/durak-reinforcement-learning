"""Bitmask game state for Numba-accelerated Durak.

State representation (all plain integers / numpy scalars):
    hands[player]   np.int64 bitmask — bit i set = card i in hand
    table_atk       np.int64 bitmask — attack cards currently on table
    table_def       np.int64 bitmask — defense cards currently on table
    table_def_for   np.ndarray[36, int8] — table_def_for[atk_cid] = def_cid (-1 if uncovered)
    discard         np.int64 bitmask — all discarded cards
    trump_suit      np.int8  (0-3)
    phase           np.int8  (0=ATTACK, 1=DEFEND)
    attacker_id     np.int8
    defender_id     np.int8
    num_players     np.int8
    first_defense_completed  np.int8  (0 or 1)
    winner          np.int8  (-1 = no winner yet)

Uncovered attack cards = table_atk & ~table_def
"""

import random

import numpy as np

from engine.fast_card import (
    NUM_CARDS,
    beats,
    card_rank,
    card_suit,
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
        rng = random.Random(seed)

        # Build and shuffle deck as card_ids 0..35
        deck = list(range(NUM_CARDS))
        rng.shuffle(deck)

        # Trump is the first card (drawn last)
        self.trump_suit = np.int8(card_suit(deck[0]))

        # Deal HAND_SIZE cards to each player into bitmask hands
        self.hands = np.zeros(num_players, dtype=np.int64)
        pos = NUM_CARDS - 1  # draw from back (pop() order)
        for player in range(num_players):
            for _ in range(HAND_SIZE):
                self.hands[player] = hand_add(int(self.hands[player]), deck[pos])
                pos -= 1

        # Remaining deck cards (after dealing)
        self.deck: list[int] = deck[:pos + 1]

        # First attacker: player with lowest trump card, random if none
        lowest_rank = 999
        attacker = -1
        for player in range(num_players):
            for cid in hand_to_list(int(self.hands[player])):
                if card_suit(cid) == self.trump_suit and card_rank(cid) < lowest_rank:
                    lowest_rank = card_rank(cid)
                    attacker = player
        if attacker == -1:
            attacker = rng.randint(0, num_players - 1)

        self.num_players = np.int8(num_players)
        self.attacker_id = np.int8(attacker)
        self.defender_id = np.int8((attacker + 1) % num_players)
        self.phase = PHASE_ATTACK
        self.table_atk = np.int64(0)
        self.table_def = np.int64(0)
        self.table_def_for = np.full(NUM_CARDS, -1, dtype=np.int8)
        self.discard = np.int64(0)
        self.first_defense_completed = np.int8(0)
        self.winner = NO_WINNER

    @property
    def uncovered_atk(self) -> int:
        """Bitmask of attack card_ids not yet covered.

        table_atk and table_def use different bit positions (different card IDs),
        so they cannot be directly ANDed. Instead, iterate attack cards and check
        the pairing array.
        """
        result = 0
        for cid in hand_to_list(int(self.table_atk)):
            if self.table_def_for[cid] == -1:
                result = hand_add(result, cid)
        return result

    @property
    def table_ranks(self) -> int:
        """Rank bitmask: bit (rank-6) set if that rank is on the table.

        Check membership with: table_ranks & (1 << (rank - 6))
        """
        rank_bits = 0
        for cid in hand_to_list(int(self.table_atk) | int(self.table_def)):
            rank_bits |= 1 << (card_rank(cid) - 6)
        return rank_bits

    @property
    def throw_in_limit(self) -> int:
        """Max attack cards allowed on table this bout."""
        if not self.first_defense_completed:
            return 5
        return hand_size(int(self.hands[self.defender_id]))


class FastGame:
    """Thin action layer over FastGameState (mirrors Game)."""

    def __init__(self, num_players: int = 2, seed: int | None = None):
        self.state = FastGameState(num_players, seed)

    def get_legal_actions(self) -> list:
        """Return legal actions as (action_str, *card_ids)."""
        s = self.state
        if s.phase == PHASE_ATTACK:
            if not s.table_atk:
                return [("attack", cid) for cid in hand_to_list(int(s.hands[s.attacker_id]))]
            else:
                legal = []
                for cid in hand_to_list(int(s.hands[s.attacker_id])):
                    if s.table_ranks & (1 << (card_rank(cid) - 6)) and hand_size(int(s.table_atk)) < s.throw_in_limit:
                        legal.append(("throw_in", cid))
                legal.append(("stop", -1))
                return legal
        else:  # DEFEND
            legal = []
            for atk_cid in hand_to_list(s.uncovered_atk):
                for def_cid in hand_to_list(int(s.hands[s.defender_id])):
                    if beats(def_cid, atk_cid, int(s.trump_suit)):
                        legal.append(("defend", atk_cid, def_cid))
            legal.append(("pick_up", -1))
            return legal

    def attack(self, cid: int) -> None:
        """Move card from attacker's hand to table_atk."""
        if self.state.phase != PHASE_ATTACK:
            raise ValueError("Cannot attack during DEFEND phase.")
        s = self.state
        s.hands[s.attacker_id] = hand_remove(int(s.hands[s.attacker_id]), cid)
        s.table_atk = np.int64(hand_add(int(s.table_atk), cid))
        s.phase = PHASE_DEFEND
        self._check_win()

    def defend(self, atk_cid: int, def_cid: int) -> None:
        """Cover atk_cid with def_cid."""
        s = self.state
        if not beats(def_cid, atk_cid, int(s.trump_suit)):
            raise ValueError(f"Defense card {def_cid} does not beat attack card {atk_cid}.")
        s.hands[s.defender_id] = hand_remove(int(s.hands[s.defender_id]), def_cid)
        s.table_def = np.int64(hand_add(int(s.table_def), def_cid))
        s.table_def_for[atk_cid] = def_cid
        s.phase = PHASE_ATTACK
        self._check_win()

    def throw_in(self, cid: int) -> None:
        """Throw in a card matching a rank already on the table."""
        s = self.state
        if s.phase != PHASE_ATTACK:
            raise ValueError("Cannot throw in during DEFEND phase.")
        if not (s.table_ranks & (1 << (card_rank(cid) - 6))):
            raise ValueError("Card rank must match a rank on the table.")
        if hand_size(int(s.table_atk)) >= s.throw_in_limit:
            raise ValueError("Throw-in limit reached.")
        s.hands[s.attacker_id] = hand_remove(int(s.hands[s.attacker_id]), cid)
        s.table_atk = np.int64(hand_add(int(s.table_atk), cid))
        s.phase = PHASE_DEFEND
        self._check_win()

    def pick_up(self) -> None:
        """Defender picks up all table cards.

        Defender who picked up is skipped — defender+1 attacks, defender+2 defends.
        """
        s = self.state
        all_table = int(s.table_atk) | int(s.table_def)
        for cid in hand_to_list(all_table):
            s.hands[s.defender_id] = hand_add(int(s.hands[s.defender_id]), cid)

        s.table_atk = np.int64(0)
        s.table_def = np.int64(0)
        s.table_def_for[:] = -1
        s.phase = PHASE_ATTACK

        self._draw_phase()

        s.attacker_id = np.int8((int(s.defender_id) + 1) % int(s.num_players))
        s.defender_id = np.int8((int(s.defender_id) + 2) % int(s.num_players))

    def stop(self) -> None:
        """Attacker ends bout after successful defense.

        Defender becomes new attacker, defender+1 becomes new defender.
        """
        s = self.state
        if s.uncovered_atk:
            raise ValueError("Cannot stop — not all attack cards are covered.")

        all_table = int(s.table_atk) | int(s.table_def)
        for cid in hand_to_list(all_table):
            s.discard = np.int64(hand_add(int(s.discard), cid))

        s.table_atk = np.int64(0)
        s.table_def = np.int64(0)
        s.table_def_for[:] = -1
        s.first_defense_completed = np.int8(1)

        self._draw_phase()
        s.phase = PHASE_ATTACK

        s.attacker_id = s.defender_id
        s.defender_id = np.int8((int(s.defender_id) + 1) % int(s.num_players))

    def _draw_phase(self) -> None:
        """Draw up to HAND_SIZE for attacker then defender."""
        s = self.state
        for player in [int(s.attacker_id), int(s.defender_id)]:
            while hand_size(int(s.hands[player])) < HAND_SIZE and s.deck:
                s.hands[player] = hand_add(int(s.hands[player]), s.deck.pop())

    def _check_win(self) -> None:
        """Set winner if any player's hand is empty and deck is empty."""
        s = self.state
        if s.deck:
            return
        for i in range(int(s.num_players)):
            if hand_size(int(s.hands[i])) == 0:
                s.winner = np.int8(i)
                break
