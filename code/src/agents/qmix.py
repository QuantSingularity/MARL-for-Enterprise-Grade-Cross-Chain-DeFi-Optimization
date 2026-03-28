"""
QMIX: Monotonic Value Function Factorization for Deep Multi-Agent RL

Based on: Rashid et al. "QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent Reinforcement Learning" (ICML 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class AgentNetwork(nn.Module):
    """Individual agent Q-network (DRQN)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, n_actions: int, n_layers: int = 2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # GRU for temporal processing
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Fully connected layers
        layers = []
        for i in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, n_actions))

        self.fc = nn.Sequential(*layers)

    def forward(
        self, obs: torch.Tensor, hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: (batch, seq_len, input_dim) or (batch, input_dim)
            hidden_state: (1, batch, hidden_dim)

        Returns:
            q_values: (batch, seq_len, n_actions) or (batch, n_actions)
            new_hidden_state: (1, batch, hidden_dim)
        """
        # Add sequence dimension if needed
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        # GRU forward
        if hidden_state is None:
            gru_out, new_hidden = self.gru(obs)
        else:
            gru_out, new_hidden = self.gru(obs, hidden_state)

        # FC layers
        q_values = self.fc(gru_out)

        if squeeze_output:
            q_values = q_values.squeeze(1)

        return q_values, new_hidden


class QMIXMixer(nn.Module):
    """
    QMIX mixing network that combines individual Q-values into Q_tot.

    Uses hypernetworks to generate weights for the mixing network,
    ensuring monotonicity constraint.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Hypernetwork for first layer weights (n_agents x hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * hidden_dim),
        )

        # Hypernetwork for first layer bias
        self.hyper_b1 = nn.Sequential(nn.Linear(state_dim, hidden_dim))

        # Hypernetwork for second layer weights (hidden_dim x 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, hidden_dim),
        )

        # Hypernetwork for second layer bias
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, 1), nn.ReLU())

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values into Q_tot.

        Args:
            q_values: (batch, n_agents)
            state: (batch, state_dim)

        Returns:
            q_tot: (batch, 1)
        """
        batch_size = q_values.size(0)
        q_values = q_values.view(batch_size, 1, self.n_agents)

        # Generate weights (ensure positivity with abs)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)

        b1 = self.hyper_b1(state)
        b1 = b1.view(batch_size, 1, self.hidden_dim)

        # First layer
        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, self.hidden_dim, 1)

        b2 = self.hyper_b2(state)
        b2 = b2.view(batch_size, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.view(batch_size, 1)

        return q_tot


class QMIXAgent:
    """
    QMIX agent for cooperative multi-agent reinforcement learning.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        lr: float = 5e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        """
        Initialize QMIX agent.

        Args:
            n_agents: Number of agents
            obs_dim: Observation dimension per agent
            state_dim: Global state dimension
            n_actions: Number of actions per agent
            hidden_dim: Hidden dimension for networks
            gamma: Discount factor
            lr: Learning rate
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Epsilon decay rate
            device: Device to use ('cpu' or 'cuda')
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)

        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Create agent networks (one per agent)
        self.agent_networks = nn.ModuleList(
            [
                AgentNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
                for _ in range(n_agents)
            ]
        )

        # Target agent networks
        self.target_agent_networks = nn.ModuleList(
            [
                AgentNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
                for _ in range(n_agents)
            ]
        )

        # Mixer network
        self.mixer = QMIXMixer(n_agents, state_dim).to(self.device)
        self.target_mixer = QMIXMixer(n_agents, state_dim).to(self.device)

        # Optimizer
        self.params = list(self.mixer.parameters())
        for net in self.agent_networks:
            self.params += list(net.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

        # Initialize target networks
        self.update_target_networks(tau=1.0)

        # Hidden states for recurrent processing
        self.hidden_states = None
        self.reset_hidden_states()

    def reset_hidden_states(self, batch_size: int = 1):
        """Reset hidden states for all agents."""
        self.hidden_states = [
            torch.zeros(1, batch_size, net.hidden_dim).to(self.device)
            for net in self.agent_networks
        ]

    def select_actions(
        self, observations: List[np.ndarray], epsilon: Optional[float] = None
    ) -> List[int]:
        """
        Select actions for all agents using epsilon-greedy.

        Args:
            observations: List of observations for each agent
            epsilon: Override epsilon value (uses self.epsilon if None)

        Returns:
            List of selected actions
        """
        if epsilon is None:
            epsilon = self.epsilon

        actions = []

        for i, obs in enumerate(observations):
            if np.random.random() < epsilon:
                # Random action
                action = np.random.randint(self.n_actions)
            else:
                # Greedy action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values, self.hidden_states[i] = self.agent_networks[i](
                        obs_tensor, self.hidden_states[i]
                    )
                action = q_values.argmax(dim=-1).item()

            actions.append(action)

        return actions

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update agent networks using a batch of experiences.

        Args:
            batch: Dictionary containing:
                - observations: (batch, n_agents, obs_dim)
                - actions: (batch, n_agents)
                - rewards: (batch, 1)
                - next_observations: (batch, n_agents, obs_dim)
                - states: (batch, state_dim)
                - next_states: (batch, state_dim)
                - dones: (batch, 1)

        Returns:
            Dictionary of training metrics
        """
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        states = batch["states"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        batch_size = observations.size(0)  # noqa: F841

        # Compute Q-values for current state
        q_values_list = []
        for i in range(self.n_agents):
            q, _ = self.agent_networks[i](observations[:, i, :])
            # Select Q-value for taken action
            q_taken = q.gather(1, actions[:, i : i + 1])
            q_values_list.append(q_taken)

        q_values = torch.cat(q_values_list, dim=1)
        q_tot = self.mixer(q_values, states)

        # Compute target Q-values for next state
        with torch.no_grad():
            target_q_values_list = []
            for i in range(self.n_agents):
                q, _ = self.target_agent_networks[i](next_observations[:, i, :])
                # Select max Q-value
                q_max = q.max(dim=1, keepdim=True)[0]
                target_q_values_list.append(q_max)

            target_q_values = torch.cat(target_q_values_list, dim=1)
            q_tot_target = self.target_mixer(target_q_values, next_states)

            # Bellman target
            target = rewards + self.gamma * (1 - dones) * q_tot_target

        # Loss
        loss = F.mse_loss(q_tot, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss.item(),
            "q_tot_mean": q_tot.mean().item(),
            "target_mean": target.mean().item(),
            "epsilon": self.epsilon,
        }

    def update_target_networks(self, tau: float = 0.01):
        """Soft update target networks."""
        # Update agent networks
        for target_net, net in zip(self.target_agent_networks, self.agent_networks):
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

        # Update mixer
        for target_param, param in zip(
            self.target_mixer.parameters(), self.mixer.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str):
        """Save agent state."""
        torch.save(
            {
                "agent_networks": [net.state_dict() for net in self.agent_networks],
                "mixer": self.mixer.state_dict(),
                "epsilon": self.epsilon,
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, net in enumerate(self.agent_networks):
            net.load_state_dict(checkpoint["agent_networks"][i])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.epsilon = checkpoint["epsilon"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.update_target_networks(tau=1.0)
