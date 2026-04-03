import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    # TODO(human): choose your architecture — hidden_dim, num_layers, activation.
    # Starting point: 2 hidden layers of 256 with ReLU.
    # Wider networks learn more complex strategies but train slower.
    def __init__(self, obs_dim: int = 150, action_dim: int = 38, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int = 150,
        action_dim: int = 38,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        # TODO(human): tune lr, gamma. lr=1e-3 is aggressive; 1e-4 is safer.
        self.gamma = gamma
        self.device = device

        self.q_online = QNetwork(obs_dim, action_dim).to(device)
        self.q_target = QNetwork(obs_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=lr)

    def select_action(self, obs: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection with illegal action masking."""
        if np.random.random() < epsilon:
            return int(np.random.choice(np.where(mask)[0]))
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q_online(obs_t).squeeze(0).cpu().numpy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def update(self, batch: tuple) -> float:
        """One DQN gradient step. Returns scalar loss."""
        obs, actions, rewards, next_obs, dones = batch
        obs_t = torch.FloatTensor(obs).to(self.device)
        act_t = torch.LongTensor(actions).to(self.device)
        rew_t = torch.FloatTensor(rewards).to(self.device)
        nobs_t = torch.FloatTensor(next_obs).to(self.device)
        done_t = torch.FloatTensor(dones).to(self.device)

        q_vals = self.q_online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(nobs_t).max(1)[0]
            targets = rew_t + self.gamma * next_q * (1 - done_t)

        loss = nn.functional.mse_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self) -> None:
        """Copy online weights to target network."""
        self.q_target.load_state_dict(self.q_online.state_dict())
