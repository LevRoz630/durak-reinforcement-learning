from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3


class Rank(IntEnum):
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    def beats(self, other: "Card", trump_suit: Suit) -> bool:
        """Can this card beat `other`?

        Rules:
        - Same suit: higher rank wins
        - Trump beats any non-trump
        - Trump vs trump: higher rank wins
        """
        if self.suit == other.suit:
            return self.rank > other.rank
        elif self.suit == trump_suit and other.suit != trump_suit:
            return True
        elif self.suit != trump_suit and other.suit == trump_suit:
            return False
        else:
            return False

    def __str__(self) -> str:
        return f"{self.rank.name.title()} of {self.suit.name.title()}"
