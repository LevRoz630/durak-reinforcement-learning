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
        """
        hands = []
        
        for i in range(num_players):
            hand = []
            self.draw(hand, HAND_SIZE)
            hands.append(hand)


        lowest_trump = (None, None)  # (lowest trump card, hand index)
        
        for i, hand in enumerate(hands):
            hand_trumps = [card for card in hand if card.suit == self.trump_suit]
            min_trump = min(hand_trumps, key=lambda c: c.rank, default=None)
            if min_trump is not None:
                if lowest_trump[0] is None or min_trump.rank < lowest_trump[0].rank:
                    lowest_trump=(min_trump, i)
                    
        return hands, lowest_trump[1] if lowest_trump[0] is not None else self.rng.randint(0, num_players - 1)

    def draw(self, hand: list[Card], count: int) -> None:
        """Draw up to `count` cards from the deck into `hand`."""
        for _ in range(count):
            if not self.cards:
                break
            hand.append(self.cards.pop(0))
