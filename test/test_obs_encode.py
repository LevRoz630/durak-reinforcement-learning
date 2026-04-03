import unittest
import numpy as np
from engine.fast_game_state import FastGame, PHASE_ATTACK, PHASE_DEFEND
from engine.fast_card import hand_size


class TestEncodeObservation(unittest.TestCase):
    def test_shape(self):
        from engine.obs_encode import encode_observation
        from engine.fast_game_state import FastGameState
        s = FastGameState(num_players=2, seed=7)
        obs = encode_observation(
            int(s.hands[0]), int(s.table_atk), int(s.table_def),
            int(s.discard), int(s.trump_suit),
            hand_size(int(s.hands[1])), int(s.phase),
        )
        self.assertEqual(obs.shape, (150,))
        self.assertEqual(obs.dtype, np.float32)

    def test_own_hand_bits(self):
        from engine.obs_encode import encode_observation
        from engine.fast_game_state import FastGameState
        from engine.fast_card import hand_to_list
        s = FastGameState(num_players=2, seed=7)
        obs = encode_observation(
            int(s.hands[0]), int(s.table_atk), int(s.table_def),
            int(s.discard), int(s.trump_suit),
            hand_size(int(s.hands[1])), int(s.phase),
        )
        for cid in hand_to_list(int(s.hands[0])):
            self.assertEqual(obs[cid], 1.0, f"card {cid} should be in hand")

    def test_empty_table_zero(self):
        from engine.obs_encode import encode_observation
        from engine.fast_game_state import FastGameState
        s = FastGameState(num_players=2, seed=7)
        obs = encode_observation(
            int(s.hands[0]), int(s.table_atk), int(s.table_def),
            int(s.discard), int(s.trump_suit),
            hand_size(int(s.hands[1])), int(s.phase),
        )
        self.assertTrue(np.all(obs[36:72] == 0.0))

    def test_trump_suit_one_hot(self):
        from engine.obs_encode import encode_observation
        from engine.fast_game_state import FastGameState
        s = FastGameState(num_players=2, seed=7)
        obs = encode_observation(
            int(s.hands[0]), int(s.table_atk), int(s.table_def),
            int(s.discard), int(s.trump_suit),
            hand_size(int(s.hands[1])), int(s.phase),
        )
        trump_slice = obs[144:148]
        self.assertEqual(trump_slice.sum(), 1.0)
        self.assertEqual(trump_slice[int(s.trump_suit)], 1.0)


class TestLegalActionMask(unittest.TestCase):
    def test_attack_phase_empty_table(self):
        from engine.obs_encode import legal_action_mask
        from engine.fast_game_state import FastGameState
        from engine.fast_card import hand_to_list, hand_size
        s = FastGameState(num_players=2, seed=7)
        mask = legal_action_mask(
            int(s.hands[s.attacker_id]), int(s.table_atk),
            s.table_def_for, int(s.trump_suit), int(s.phase),
            s.table_ranks, hand_size(int(s.table_atk)), s.throw_in_limit,
        )
        self.assertEqual(mask.shape, (38,))
        for cid in hand_to_list(int(s.hands[s.attacker_id])):
            self.assertTrue(mask[cid], f"card {cid} should be legal to attack")
        self.assertFalse(mask[36])  # stop
        self.assertFalse(mask[37])  # pick_up

    def test_defend_phase_pick_up_always_legal(self):
        from engine.obs_encode import legal_action_mask
        from engine.fast_game_state import FastGame, PHASE_DEFEND
        from engine.fast_card import hand_size
        game = FastGame(num_players=2, seed=7)
        actions = game.get_legal_actions()
        atk = next(a for a in actions if a[0] == "attack")
        game.attack(atk[1])
        s = game.state
        mask = legal_action_mask(
            int(s.hands[s.defender_id]), int(s.table_atk),
            s.table_def_for, int(s.trump_suit), int(s.phase),
            s.table_ranks, hand_size(int(s.table_atk)), s.throw_in_limit,
        )
        self.assertTrue(mask[37])  # pick_up always legal in DEFEND
        self.assertFalse(mask[36])  # stop not legal in DEFEND


if __name__ == "__main__":
    unittest.main(verbosity=2)
