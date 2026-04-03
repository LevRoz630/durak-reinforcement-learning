from enum import IntEnum

from engine.legacy.card import Card, Suit
from engine.legacy.deck import Deck, HAND_SIZE


class Phase(IntEnum):
    ATTACK = 0
    DEFEND = 1


class GameState:
    def __init__(self, num_players: int = 2, seed: int | None = None):
        import random

        rng = random.Random(seed)
        self.deck = Deck(rng)
        self.trump_suit: Suit = self.deck.trump_suit
        self.hands: list[list[Card]]
        self.attacker_id: int
        self.hands, self.attacker_id = self.deck.deal(num_players)
        self.defender_id: int = (self.attacker_id + 1) % num_players
        self.num_players: int = num_players
        self.table: list[tuple[Card, Card | None]] = []
        self.discard_pile: list[Card] = []
        self.phase: Phase = Phase.ATTACK
        self.first_defense_completed: bool = False
        self.winner: int | None = None

    @property
    def throw_in_limit(self) -> int:
        """Max attack cards allowed on the table this bout."""
        if not self.first_defense_completed:
            return 5
        return len(self.hands[self.defender_id])

    @property
    def uncovered_cards(self) -> list[Card]:
        """Attack cards on the table that haven't been covered yet."""
        return [atk for atk, dfn in self.table if dfn is None]

    @property
    def table_ranks(self) -> set:
        """All ranks currently on the table (for validating throw-ins)."""
        ranks = set()
        for atk, dfn in self.table:
            ranks.add(atk.rank)
            if dfn is not None:
                ranks.add(dfn.rank)
        return ranks
