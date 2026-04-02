"""Integer-encoded card primitives, Numba-compatible.

Card encoding:
    card_id = suit * 9 + (rank - 6)   →   0..35
    suit    = card_id // 9             →   0=Hearts, 1=Diamonds, 2=Clubs, 3=Spades
    rank    = card_id % 9 + 6         →   6..14

All functions are pure integer arithmetic so they can be decorated with @njit.
"""

NUM_CARDS = 36
NUM_SUITS = 4
NUM_RANKS = 9  # 6..Ace


def card_id(suit: int, rank: int) -> int:
    """Encode (suit 0-3, rank 6-14) → card_id 0-35."""
    return suit * NUM_RANKS + (rank - 6)


def card_suit(cid: int) -> int:
    """Recover suit from card_id."""
    return cid // NUM_RANKS


def card_rank(cid: int) -> int:
    """Recover rank (6-14) from card_id."""
    return cid % NUM_RANKS + 6


def beats(cid_a: int, cid_b: int, trump_suit: int) -> bool:
    """Return True if card_a can beat card_b given trump_suit.

    - same suit: higher rank wins
    - trump beats any non-trump
    - trump vs trump: higher rank wins
    - different non-trump suits: False
    """
    if card_suit(cid_a) == card_suit(cid_b):
        return card_rank(cid_a) > card_rank(cid_b)
    elif card_suit(cid_a) == trump_suit and card_suit(cid_b) != trump_suit:
        return True
    else:
        return False


def hand_contains(hand: int, cid: int) -> bool:
    """Return True if card cid is present in the bitmask hand."""
    return bool(hand & (1 << cid))


def hand_add(hand: int, cid: int) -> int:
    """Return new hand bitmask with cid added."""
    return hand | (1 << cid)


def hand_remove(hand: int, cid: int) -> int:
    """Return new hand bitmask with cid removed."""
    return hand & ~(1 << cid)


def hand_size(hand: int) -> int:
    """Return number of cards in the bitmask hand.

    Uses the Kernighan bit-clearing trick: subtracting 1 from any number
    flips the lowest set bit to 0 and all zeros below it to 1.
    AND-ing with the original clears exactly that lowest set bit, leaving
    everything above it untouched. The loop therefore runs once per set bit,
    skipping zeros entirely — O(cards in hand) not O(total bits).

        hand     = 0b00001010
        hand - 1 = 0b00001001  (borrowed through bit 1)
        AND      = 0b00001000  (bit 1 cleared, bit 3 survives)
        → repeat → 0b00000000, count = 2
    """
    count = 0
    while hand:
        hand &= hand - 1
        count += 1
    return count


def hand_to_list(hand: int) -> list[int]:
    """Expand bitmask hand to a list of card_ids (for iteration / debug)."""
    result = []
    for cid in range(NUM_CARDS):
        if hand_contains(hand, cid):
            result.append(cid)
    return result
