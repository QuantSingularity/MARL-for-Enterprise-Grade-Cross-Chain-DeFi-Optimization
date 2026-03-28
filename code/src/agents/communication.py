"""
Agent Communication Module for MARL

Implements communication protocols between agents including:
- Message passing
- Attention-based communication
- CommNet-style architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union


class CommNet(nn.Module):
    """
    CommNet: Learning Multiagent Communication with Backpropagation

    Implements iterative message passing between agents.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, n_agents: int, n_comm_rounds: int = 2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        self.n_comm_rounds = n_comm_rounds

        # Encoding layer
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # Communication module
        self.comm_layer = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with communication.

        Args:
            observations: (batch, n_agents, input_dim)

        Returns:
            hidden_states: (batch, n_agents, hidden_dim)
        """
        batch_size = observations.size(0)  # noqa: F841

        # Encode observations
        hidden = F.relu(self.encoder(observations))

        # Communication rounds
        for _ in range(self.n_comm_rounds):
            # Average messages from all agents
            messages = hidden.mean(dim=1, keepdim=True).expand(-1, self.n_agents, -1)

            # Update hidden states
            hidden = F.relu(self.comm_layer(messages) + hidden)

        # Output
        output = self.output_layer(hidden)

        return output


class AttentionCommunicationModule(nn.Module):
    """
    Attention-based Communication.

    Uses multi-head attention for selective message passing.
    Projects raw observations (input_dim) to hidden_dim before communication,
    and supports returning attention weights for analysis.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_agents: int,
        n_heads: int = 4,
        # Accept both 'obs_dim' (test-style) and 'input_dim' via constructor
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        self.n_heads = n_heads

        self.encoder = nn.Linear(input_dim, hidden_dim)

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        observations: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with attention-based communication.

        Args:
            observations: (batch, n_agents, input_dim)
            return_attention: If True, also return attention weight tensor

        Returns:
            updated_states: (batch, n_agents, hidden_dim)
            attn_weights:   (batch, n_agents, n_agents)  — only if return_attention=True
        """
        # Project observations to hidden_dim
        hidden_states = F.relu(self.encoder(observations))

        # Self-attention
        attn_out, attn_weights = self.attention(
            hidden_states, hidden_states, hidden_states
        )

        # Residual connection + layer norm
        hidden_states = self.layer_norm1(hidden_states + attn_out)

        # Feed-forward + residual
        ff_out = self.ff(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + ff_out)

        if return_attention:
            return hidden_states, attn_weights
        return hidden_states


# Backward-compatible alias
AttentionComm = AttentionCommunicationModule


class MessagePool:
    """
    Centralized message pool for agent communication.

    Allows agents to post and retrieve messages.
    """

    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self.messages = []

    def post(self, sender_id: int, message: torch.Tensor, metadata: dict = None):
        """Post a message to the pool."""
        self.messages.append(
            {"sender": sender_id, "content": message, "metadata": metadata or {}}
        )

        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def retrieve(self, agent_id: int, filter_fn=None) -> List[dict]:
        """Retrieve messages for an agent."""
        if filter_fn is None:
            return self.messages
        return [msg for msg in self.messages if filter_fn(msg, agent_id)]

    def clear(self):
        """Clear all messages."""
        self.messages = []
