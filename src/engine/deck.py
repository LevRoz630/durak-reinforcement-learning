import random

from engine.card import Card, Rank, Suit

ALL_CARDS = [Card(suit, rank) for suit in Suit for rank in Rank]
HAND_SIZE = 6


class Deck:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.cards: list[Card] = list(ALL_CARDS)
        self.rng.shuffle(self.cards)
        self.trump_suit: Suit = self.cards[-1].suit

    def deal(self, num_players: int) -> tuple[list[list[Card]], int]:
        """Deal HAND_SIZE cards to each player and determine first attacker.

        Returns:
            (hands, first_attacker_id)

        First attacker: player with the lowest trump card.
        If no player has a trump, pick randomly.

        TODO(human): Implement the dealing and first-attacker logic.
        """
        raise NotImplementedError

    def draw(self, hand: list[Card], count: int) -> None:
        """Draw up to `count` cards from the deck into `hand`."""
        for _ in range(count):
            if not self.cards:
                break
            hand.append(self.cards.pop(0))
