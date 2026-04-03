# Engine Implementation Plan (Python)

Podkidnoy (throw-in) Durak, 2-player (extensible to n-player).

## Data Model

### Card (`src/engine/card.py`)
- `Suit(IntEnum)`: HEARTS=0, DIAMONDS=1, CLUBS=2, SPADES=3
- `Rank(IntEnum)`: SIX=6 through ACE=14
- `Card` dataclass (frozen): suit, rank
- `beats(other, trump_suit)` method — same suit higher rank, or trump beats non-trump

### Deck (`src/engine/deck.py`)
- All 36 cards (4 suits x 9 ranks)
- Shuffle, deal, draw

### GameState (`src/engine/game_state.py`)
- `hands: list[list[Card]]` — indexed by player id (0, 1, ..., n)
- `deck: list[Card]`
- `trump_suit: Suit`
- `table: list[tuple[Card, Card | None]]` — ordered attack/defense pairs
- `attacker_id: int`
- `defender_id: int`
- `phase: Phase` — ATTACK / DEFEND
- `first_defense_completed: bool`

### Game Logic (`src/engine/game.py`)
- Legal actions per phase
- Play card, cover card, throw in, stop, pick up
- Bout resolution, draw phase, role swap
- Win detection (empty hand = instant win)

## Rules

- 36-card deck, 6 cards dealt to each player
- Lowest trump attacks first; no trumps = random
- Beat: same suit + higher rank, or trump beats non-trump
- Throw-in limit: 5 before first successful defense, then defender's hand size
- Throw-in cards must match a rank already on the table
- Draw: attacker first, up to 6 cards, from remaining deck
- Roles swap after bout (unless defender picked up)
- Empty hand at any point = instant win

## Build Order

1. Card, Suit, Rank (data model)
2. Deck (shuffle, deal)
3. GameState (fields + initialization)
4. Game logic (legal actions, play, cover, throw in, bout resolution, draw)
