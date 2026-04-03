import unittest

from engine.game import Game
from engine.game_state import Phase
from engine.card import Card

# hand should be shown to player before actions have to be decided.
def validate_state(game: Game) -> None:
    """Basic invariants for a 2-player Durak game."""
    state = game.state

    assert state.num_players == 2, "Expected exactly 2 players."
    assert state.attacker_id in (0, 1), "Invalid attacker_id."
    assert state.defender_id in (0, 1), "Invalid defender_id."
    assert state.attacker_id != state.defender_id, "Attacker and defender must differ."

    all_hand_cards = [card for hand in state.hands for card in hand]
    table_cards = []
    for atk, dfn in state.table:
        table_cards.append(atk)
        if dfn is not None:
            table_cards.append(dfn)

    all_cards = all_hand_cards + table_cards + list(state.deck.cards) + list(state.discard_pile)
    assert len(all_cards) == 36, f"Card count mismatch: expected 36, got {len(all_cards)}."
    assert len(set(all_cards)) == 36, "Duplicate or missing cards detected."

    uncovered = [atk for atk, dfn in state.table if dfn is None]
    assert uncovered == state.uncovered_cards, "uncovered_cards property is inconsistent."

    if state.phase == Phase.DEFEND and state.table:
        assert len(state.uncovered_cards) >= 1, "DEFEND phase should have an uncovered attack card."


def choose_action(game: Game):
    """Pick a deterministic legal action to keep the simulation reproducible."""
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


def apply_action(game: Game, action):
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
    else:
        raise RuntimeError(f"Unknown action: {action}")


class TestTwoPlayerDurak(unittest.TestCase):
    def test_game_initializes(self):
        game = Game(num_players=2, seed=7)
        state = game.state

        self.assertEqual(state.num_players, 2)
        self.assertEqual(len(state.hands), 2)
        self.assertEqual(len(state.hands[0]), 6)
        self.assertEqual(len(state.hands[1]), 6)
        self.assertEqual(len(state.deck.cards), 24)
        self.assertIn(state.attacker_id, [0, 1])
        self.assertIn(state.defender_id, [0, 1])
        self.assertNotEqual(state.attacker_id, state.defender_id)
        self.assertEqual(state.phase, Phase.ATTACK)

        validate_state(game)

    def test_attack_then_defend_or_pickup(self):
        game = Game(num_players=2, seed=7)

        action = choose_action(game)
        self.assertEqual(action[0], "attack")
        game.attack(action[1])

        self.assertEqual(game.state.phase, Phase.DEFEND)
        self.assertEqual(len(game.state.table), 1)
        self.assertIsNone(game.state.table[0][1])

        legal = game.get_legal_actions()
        self.assertTrue(any(a[0] in ("defend", "pick_up") for a in legal))
        validate_state(game)

    def test_simulate_game(self):
        game = Game(num_players=2, seed=7)

        max_steps = 500
        steps = 0

        while game.state.winner is None and steps < max_steps:
            validate_state(game)
            legal = game.get_legal_actions()
            self.assertTrue(len(legal) > 0, "No legal actions available before game end.")
            action = choose_action(game)
            apply_action(game, action)
            steps += 1

        validate_state(game)

        self.assertIsNotNone(
            game.state.winner,
            f"Game did not finish within {max_steps} steps."
        )
        self.assertIn(game.state.winner, [0, 1])

    def test_multiple_seeds(self):
        for seed in range(10):
            game = Game(num_players=2, seed=seed)
            max_steps = 500
            steps = 0

            while game.state.winner is None and steps < max_steps:
                validate_state(game)
                action = choose_action(game)
                apply_action(game, action)
                steps += 1

            self.assertIsNotNone(
                game.state.winner,
                f"Game with seed {seed} did not finish."
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
