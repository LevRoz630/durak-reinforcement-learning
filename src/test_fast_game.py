"""Cross-validate FastGame against the reference Game implementation.

Strategy: run both games with the same seed using the same deterministic policy.
Actions chosen from Game (Card objects) are translated to card_ids and applied
to FastGame. At each step we assert card accounting holds. At the end we assert
both games produce the same winner.
"""

import unittest

from engine.card import Card
from engine.fast_card import NUM_CARDS, card_rank, card_suit, hand_to_list
from engine.fast_game_state import PHASE_ATTACK, PHASE_DEFEND, FastGame
from engine.game import Game
from engine.game_state import Phase


# ---------------------------------------------------------------------------
# Helpers: translate between Card objects and card_ids
# ---------------------------------------------------------------------------

def card_to_id(card: Card) -> int:
    return int(card.suit) * 9 + (int(card.rank) - 6)


def fast_hand_cards(fast_game: FastGame, player: int) -> set[int]:
    return set(hand_to_list(int(fast_game.state.hands[player])))


def fast_all_cards(fast_game: FastGame) -> set[int]:
    """All card_ids currently tracked by the fast game (hands + table + discard + deck)."""
    cards = set()
    for p in range(int(fast_game.state.num_players)):
        cards |= set(hand_to_list(int(fast_game.state.hands[p])))
    cards |= set(hand_to_list(int(fast_game.state.table_atk)))
    cards |= set(hand_to_list(int(fast_game.state.table_def)))
    cards |= set(hand_to_list(int(fast_game.state.discard)))
    cards |= set(fast_game.state.deck)
    return cards


# ---------------------------------------------------------------------------
# Deterministic policy (mirrors test_game_2players.py choose_action)
# Applied to Game, then translated for FastGame.
# ---------------------------------------------------------------------------

def choose_action(game: Game):
    actions = game.get_legal_actions()

    defend_actions = [a for a in actions if a[0] == "defend"]
    if defend_actions:
        defend_actions.sort(key=lambda a: (a[1].rank, a[2].rank, a[2].suit))
        return defend_actions[0]

    attack_actions = [a for a in actions if a[0] == "attack"]
    if attack_actions:
        attack_actions.sort(key=lambda a: (a[1].rank, a[1].suit))
        return attack_actions[0]

    throw_actions = [a for a in actions if a[0] == "throw_in"]
    if throw_actions:
        throw_actions.sort(key=lambda a: (a[1].rank, a[1].suit))
        return throw_actions[0]

    if ("stop", None) in actions:
        return ("stop", None)

    if ("pick_up", None) in actions:
        return ("pick_up", None)

    raise RuntimeError(f"No supported legal action found: {actions}")


def apply_to_game(game: Game, action) -> None:
    kind = action[0]
    if kind == "attack":
        game.attack(action[1])
    elif kind == "defend":
        game.defend(action[1], action[2])
    elif kind == "throw_in":
        game.throw_in(action[1])
    elif kind == "stop":
        game.stop()
    elif kind == "pick_up":
        game.pick_up()


def apply_to_fast(fast: FastGame, action) -> None:
    kind = action[0]
    if kind == "attack":
        fast.attack(card_to_id(action[1]))
    elif kind == "defend":
        fast.defend(card_to_id(action[1]), card_to_id(action[2]))
    elif kind == "throw_in":
        fast.throw_in(card_to_id(action[1]))
    elif kind == "stop":
        fast.stop()
    elif kind == "pick_up":
        fast.pick_up()


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

def validate_fast_state(fast: FastGame, step: int) -> None:
    """Assert card accounting: all 36 cards are tracked exactly once."""
    all_cards = fast_all_cards(fast)
    assert len(all_cards) == NUM_CARDS, (
        f"Step {step}: card count mismatch — got {len(all_cards)}, expected {NUM_CARDS}. "
        f"Cards: {sorted(all_cards)}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFastGameInit(unittest.TestCase):
    def test_card_count(self):
        fast = FastGame(num_players=2, seed=7)
        self.assertEqual(len(fast_all_cards(fast)), NUM_CARDS)

    def test_hand_sizes(self):
        fast = FastGame(num_players=2, seed=7)
        for p in range(2):
            self.assertEqual(len(fast_hand_cards(fast, p)), 6)

    def test_deck_size(self):
        fast = FastGame(num_players=2, seed=7)
        self.assertEqual(len(fast.state.deck), 24)

    def test_roles_differ(self):
        fast = FastGame(num_players=2, seed=7)
        self.assertNotEqual(int(fast.state.attacker_id), int(fast.state.defender_id))

    def test_phase_is_attack(self):
        fast = FastGame(num_players=2, seed=7)
        self.assertEqual(fast.state.phase, PHASE_ATTACK)


class TestFastVsReference(unittest.TestCase):
    def _run_parallel(self, seed: int, max_steps: int = 500):
        game = Game(num_players=2, seed=seed)
        fast = FastGame(num_players=2, seed=seed)

        # Assert same starting attacker
        self.assertEqual(game.state.attacker_id, int(fast.state.attacker_id),
                         f"seed={seed}: attacker mismatch at start")

        steps = 0
        while game.state.winner is None and steps < max_steps:
            validate_fast_state(fast, steps)

            action = choose_action(game)
            apply_to_game(game, action)
            apply_to_fast(fast, action)
            steps += 1

        self.assertIsNotNone(game.state.winner, f"seed={seed}: reference game did not finish.")
        self.assertIsNotNone(fast.state.winner, f"seed={seed}: fast game did not finish.")
        self.assertEqual(
            game.state.winner, int(fast.state.winner),
            f"seed={seed}: winner mismatch — reference={game.state.winner}, fast={fast.state.winner}"
        )

    def test_seed_7(self):
        self._run_parallel(seed=7)

    def test_multiple_seeds(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                self._run_parallel(seed=seed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
