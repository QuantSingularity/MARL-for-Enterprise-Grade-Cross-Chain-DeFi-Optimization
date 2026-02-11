"""
MAPPO: Multi-Agent Proximal Policy Optimization

Based on: Yu et al. "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games" (NeurIPS 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """Policy network for an agent."""

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

        self.feature_extractor = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: (batch, input_dim)

        Returns:
            action_logits: (batch, n_actions)
        """
        features = self.feature_extractor(obs)
        logits = self.action_head(features)
        return logits

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and entropy.

        Args:
            obs: (batch, input_dim)
            action: (batch,) optional action to evaluate

        Returns:
            action: (batch,)
            log_prob: (batch,)
            entropy: (batch,)
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy


class MAPPOCritic(nn.Module):
    """
    Centralized critic for MAPPO.

    Takes global state and all agent observations as input.
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        obs_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()

        input_dim = state_dim + n_agents * obs_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: (batch, state_dim)
            observations: (batch, n_agents, obs_dim)

        Returns:
            value: (batch, 1)
        """
        batch_size = observations.size(0)
        obs_flat = observations.view(batch_size, -1)
        x = torch.cat([state, obs_flat], dim=1)
        value = self.network(x)
        return value


class MAPPOAgent:
    """
    Multi-Agent PPO agent.

    Uses parameter sharing across agents with agent-specific observations.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ):
        """
        Initialize MAPPO agent.

        Args:
            n_agents: Number of agents
            obs_dim: Observation dimension per agent
            state_dim: Global state dimension
            n_actions: Number of actions per agent
            hidden_dim: Hidden dimension for networks
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping epsilon
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use ('cpu' or 'cuda')
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # Actor network (shared across agents)
        self.actor = ActorNetwork(obs_dim, hidden_dim, n_actions).to(self.device)

        # Centralized critic
        self.critic = MAPPOCritic(state_dim, n_agents, obs_dim, hidden_dim).to(
            self.device
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_actions(
        self, observations: List[np.ndarray], deterministic: bool = False
    ) -> List[int]:
        """
        Select actions for all agents.

        Args:
            observations: List of observations for each agent
            deterministic: If True, use greedy action selection

        Returns:
            List of selected actions
        """
        actions = []

        for obs in observations:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.actor(obs_tensor)

                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    dist = Categorical(logits=logits)
                    action = dist.sample()

            actions.append(action.item())

        return actions

    def evaluate_actions(
        self, observations: torch.Tensor, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.

        Args:
            observations: (batch, n_agents, obs_dim)
            states: (batch, state_dim)
            actions: (batch, n_agents)

        Returns:
            log_probs: (batch, n_agents)
            entropy: (batch, n_agents)
            values: (batch, 1)
        """
        observations.size(0)

        log_probs = []
        entropies = []

        for i in range(self.n_agents):
            obs_i = observations[:, i, :]
            action_i = actions[:, i]

            logits = self.actor(obs_i)
            dist = Categorical(logits=logits)

            log_prob = dist.log_prob(action_i)
            entropy = dist.entropy()

            log_probs.append(log_prob)
            entropies.append(entropy)

        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1)

        values = self.critic(states, observations)

        return log_probs, entropies, values

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: (batch,)
            values: (batch,)
            dones: (batch,)

        Returns:
            advantages: (batch,)
            returns: (batch,)
        """
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            )
            last_advantage = advantages[t]

        returns = advantages + values

        return advantages, returns

    def update(
        self, batch: Dict[str, torch.Tensor], n_epochs: int = 10
    ) -> Dict[str, float]:
        """
        Update actor and critic networks.

        Args:
            batch: Dictionary containing:
                - observations: (batch, n_agents, obs_dim)
                - actions: (batch, n_agents)
                - rewards: (batch,)
                - states: (batch, state_dim)
                - dones: (batch,)
                - old_log_probs: (batch, n_agents)
            n_epochs: Number of PPO epochs

        Returns:
            Dictionary of training metrics
        """
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        states = batch["states"].to(self.device)
        dones = batch["dones"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)

        # Compute values and advantages
        with torch.no_grad():
            values = self.critic(states, observations).squeeze(-1)

        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for _ in range(n_epochs):
            # Evaluate actions
            log_probs, entropy, new_values = self.evaluate_actions(
                observations, states, actions
            )

            # Policy loss (sum over agents, mean over batch)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(
                ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
            ) * advantages.unsqueeze(1)
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_pred = new_values.squeeze(-1)
            critic_loss = F.mse_loss(value_pred, returns)

            # Entropy bonus
            entropy_bonus = entropy.mean()

            # Total loss
            loss = (
                actor_loss
                + self.value_loss_coef * critic_loss
                - self.entropy_coef * entropy_bonus
            )

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy_bonus.item()

        n_updates = n_epochs

        return {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "advantage_mean": advantages.mean().item(),
            "value_mean": values.mean().item(),
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
