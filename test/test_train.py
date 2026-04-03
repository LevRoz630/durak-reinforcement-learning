import unittest


class TestTrainSmoke(unittest.TestCase):
    def test_runs_without_crash(self):
        """Training loop runs for a small number of steps without error."""
        from training.train import train
        stats = train(
            num_episodes=5,
            buffer_capacity=500,
            batch_size=16,
            learning_starts=32,
            target_update_freq=50,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.99,
        )
        self.assertIn("episodes", stats)
        self.assertEqual(stats["episodes"], 5)

    def test_steps_accumulate(self):
        """Training loop accumulates steps across episodes (each game takes >0 steps)."""
        from training.train import train
        stats = train(num_episodes=3, learning_starts=1000)
        self.assertGreater(stats["steps"], 3)  # at minimum 1 step per episode


if __name__ == "__main__":
    unittest.main(verbosity=2)
