import unittest
import numpy as np


class TestReplayBuffer(unittest.TestCase):
    def _make(self, capacity=100):
        from training.replay_buffer import ReplayBuffer
        return ReplayBuffer(capacity=capacity, obs_dim=150)

    def test_push_and_len(self):
        buf = self._make()
        obs = np.zeros(150, dtype=np.float32)
        buf.push(obs, 5, 1.0, obs, False)
        self.assertEqual(len(buf), 1)

    def test_circular_wrap(self):
        buf = self._make(capacity=10)
        obs = np.zeros(150, dtype=np.float32)
        for i in range(15):
            buf.push(obs, i % 38, 0.0, obs, False)
        self.assertEqual(len(buf), 10)  # capped at capacity

    def test_sample_shape(self):
        buf = self._make()
        obs = np.random.rand(150).astype(np.float32)
        for _ in range(20):
            buf.push(obs, 3, 0.0, obs, False)
        batch = buf.sample(8)
        obs_b, act_b, rew_b, nobs_b, done_b = batch
        self.assertEqual(obs_b.shape, (8, 150))
        self.assertEqual(act_b.shape, (8,))
        self.assertEqual(rew_b.shape, (8,))

    def test_sample_requires_enough_data(self):
        buf = self._make()
        obs = np.zeros(150, dtype=np.float32)
        buf.push(obs, 0, 0.0, obs, False)
        with self.assertRaises(ValueError):
            buf.sample(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
