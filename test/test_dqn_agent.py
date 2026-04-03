import unittest
import numpy as np


class TestQNetwork(unittest.TestCase):
    def test_output_shape(self):
        import torch
        from training.dqn_agent import QNetwork
        net = QNetwork(obs_dim=150, action_dim=38)
        x = torch.zeros(4, 150)  # batch of 4
        out = net(x)
        self.assertEqual(out.shape, (4, 38))


class TestDQNAgent(unittest.TestCase):
    def _make(self):
        from training.dqn_agent import DQNAgent
        return DQNAgent(obs_dim=150, action_dim=38)

    def test_select_action_respects_mask(self):
        agent = self._make()
        obs = np.zeros(150, dtype=np.float32)
        mask = np.zeros(38, dtype=bool)
        mask[7] = True  # only action 7 is legal
        action = agent.select_action(obs, mask, epsilon=0.0)
        self.assertEqual(action, 7)

    def test_select_action_random_is_legal(self):
        agent = self._make()
        obs = np.zeros(150, dtype=np.float32)
        mask = np.zeros(38, dtype=bool)
        mask[3] = mask[15] = mask[36] = True
        for _ in range(50):
            action = agent.select_action(obs, mask, epsilon=1.0)
            self.assertIn(action, [3, 15, 36])

    def test_update_returns_loss(self):
        agent = self._make()
        batch = (
            np.zeros((8, 150), dtype=np.float32),
            np.zeros(8, dtype=np.int64),
            np.zeros(8, dtype=np.float32),
            np.zeros((8, 150), dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        )
        loss = agent.update(batch)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
