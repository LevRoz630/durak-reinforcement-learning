import numpy as np


class ReplayBuffer:
    """Circular replay buffer storing (obs, action, reward, next_obs, done) tuples."""

    def __init__(self, capacity: int, obs_dim: int = 150):
        self.capacity = capacity
        self._pos = 0
        self._size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool) -> None:
        self.obs[self._pos] = obs
        self.next_obs[self._pos] = next_obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        if batch_size > self._size:
            raise ValueError(f"Not enough data: have {self._size}, need {batch_size}")
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self._size
