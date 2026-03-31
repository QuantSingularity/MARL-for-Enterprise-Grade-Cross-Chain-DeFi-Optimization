"""
Baseline Agents for Comparison

Implements simple baselines for MARL evaluation:
- RandomAgent: Random action selection
- IndependentQLearning: Independent Q-learning (no coordination)
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomAgent:
    """Agent that selects actions uniformly at random."""

    def __init__(self, n_agents: int, n_actions: int):
        """
        Initialize random agent.

        Args:
            n_agents: Number of agents
            n_actions: Number of actions per agent
        """
        self.n_agents = n_agents
        self.n_actions = n_actions

    def select_actions(self, observations: List[np.ndarray]) -> List[int]:
        """Select random actions for all agents."""
        return [np.random.randint(self.n_actions) for _ in range(self.n_agents)]

    def update(self, batch: Dict) -> Dict[str, float]:
        """No training for random agent."""
        return {"loss": 0.0}

    def save(self, path: str):
        """No state to save."""

    def load(self, path: str):
        """No state to load."""


class QNetwork(nn.Module):
    """Simple Q-network for independent Q-learning."""

    def __init__(
        self, input_dim: int, hidden_dim: int, n_actions: int, n_layers: int = 2
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class IndependentQLearning:
    """
    Independent Q-learning baseline.

    Each agent learns its own Q-function without coordination.
    This serves as a lower bound for cooperative MARL algorithms.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        """
        Initialize independent Q-learning agent.

        Args:
            n_agents: Number of agents
            obs_dim: Observation dimension per agent
            n_actions: Number of actions per agent
            hidden_dim: Hidden dimension for Q-networks
            gamma: Discount factor
            lr: Learning rate
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Epsilon decay rate
            device: Device to use
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)

        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Create Q-networks (one per agent)
        self.q_networks = [
            QNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
            for _ in range(n_agents)
        ]

        # Target networks
        self.target_networks = [
            QNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
            for _ in range(n_agents)
        ]

        # Optimizers
        self.optimizers = [
            torch.optim.Adam(net.parameters(), lr=lr) for net in self.q_networks
        ]

        # Initialize target networks
        self.update_target_networks(tau=1.0)

    def select_actions(
        self, observations: List[np.ndarray], epsilon: Optional[float] = None
    ) -> List[int]:
        """
        Select actions using epsilon-greedy.

        Args:
            observations: List of observations for each agent
            epsilon: Override epsilon value

        Returns:
            List of selected actions
        """
        if epsilon is None:
            epsilon = self.epsilon

        actions = []

        for i, obs in enumerate(observations):
            if np.random.random() < epsilon:
                action = np.random.randint(self.n_actions)
            else:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_networks[i](obs_tensor)
                action = q_values.argmax(dim=-1).item()

            actions.append(action)

        return actions

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update Q-networks independently.

        Args:
            batch: Dictionary containing:
                - observations: (batch, n_agents, obs_dim)
                - actions: (batch, n_agents)
                - rewards: (batch, n_agents)
                - next_observations: (batch, n_agents, obs_dim)
                - dones: (batch,)

        Returns:
            Dictionary of training metrics
        """
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)

        total_loss = 0

        # Update each agent independently
        for i in range(self.n_agents):
            obs_i = observations[:, i, :]
            actions_i = actions[:, i]
            rewards_i = rewards[:, i]
            next_obs_i = next_observations[:, i, :]

            # Current Q-values
            q_values = self.q_networks[i](obs_i)
            q_taken = q_values.gather(1, actions_i.unsqueeze(1)).squeeze(1)

            # Target Q-values
            with torch.no_grad():
                next_q = self.target_networks[i](next_obs_i)
                next_q_max = next_q.max(dim=1)[0]
                target = rewards_i + self.gamma * (1 - dones) * next_q_max

            # Loss
            loss = F.mse_loss(q_taken, target)

            # Optimize
            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

            total_loss += loss.item()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {"loss": total_loss / self.n_agents, "epsilon": self.epsilon}

    def update_target_networks(self, tau: float = 0.01):
        """Soft update target networks."""
        for target_net, net in zip(self.target_networks, self.q_networks):
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

    def save(self, path: str):
        """Save agent state."""
        torch.save(
            {
                "q_networks": [net.state_dict() for net in self.q_networks],
                "epsilon": self.epsilon,
                "optimizers": [opt.state_dict() for opt in self.optimizers],
            },
            path,
        )

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, net in enumerate(self.q_networks):
            net.load_state_dict(checkpoint["q_networks"][i])
        self.epsilon = checkpoint["epsilon"]
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(checkpoint["optimizers"][i])
        self.update_target_networks(tau=1.0)


class GreedyAgent:
    """
    Greedy agent that always takes the action with immediate best reward.

    This serves as a simple heuristic baseline.
    """

    def __init__(self, n_agents: int, n_actions: int):
        self.n_agents = n_agents
        self.n_actions = n_actions

    def select_actions(self, observations: List[np.ndarray]) -> List[int]:
        """Select actions (placeholder - requires environment for greedy selection)."""
        # In practice, would need access to environment transition model
        # For now, just return random actions
        return [np.random.randint(self.n_actions) for _ in range(self.n_agents)]
