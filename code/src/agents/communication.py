"""
Attention-Based Communication Module for MARL

Implements multi-head attention for inter-agent communication
with communication budget constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights


class AttentionCommunicationModule(nn.Module):
    """
    Attention-based communication module for MARL agents.

    Allows agents to communicate by attending to other agents' hidden states.
    Implements a communication budget to limit communication overhead.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        comm_budget: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize communication module.

        Args:
            n_agents: Number of agents
            obs_dim: Observation dimension
            hidden_dim: Hidden dimension for communication
            n_heads: Number of attention heads
            comm_budget: Maximum number of agents each agent can communicate with
            dropout: Dropout rate
        """
        super().__init__()

        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.comm_budget = min(comm_budget, n_agents - 1)

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim, n_heads, dropout)

        # Message processor
        self.message_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # Communication gate (learns when to communicate)
        self.comm_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(
        self, observations: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with communication.

        Args:
            observations: (batch, n_agents, obs_dim)
            return_attention: Whether to return attention weights

        Returns:
            enhanced_obs: (batch, n_agents, hidden_dim)
            attention_weights: (batch, n_agents, n_agents) if return_attention else None
        """
        batch_size = observations.size(0)

        # Encode observations
        encoded = self.obs_encoder(observations)  # (batch, n_agents, hidden_dim)

        # Compute communication gates
        gates = self.comm_gate(encoded)  # (batch, n_agents, 1)

        # Self-attention for communication
        attended, attention_weights = self.attention(
            encoded, encoded, encoded
        )  # (batch, n_agents, hidden_dim)

        # Apply top-k communication budget
        if self.comm_budget < self.n_agents - 1:
            # Mask out low-attention connections
            attention_weights_flat = attention_weights.mean(dim=1)  # Average over heads
            top_k_values, top_k_indices = torch.topk(
                attention_weights_flat.view(batch_size * self.n_agents, -1),
                k=self.comm_budget + 1,  # +1 for self-attention
                dim=-1,
            )

            # Create mask
            mask = torch.zeros_like(attention_weights_flat)
            for b in range(batch_size):
                for i in range(self.n_agents):
                    mask[b, i, top_k_indices[b * self.n_agents + i]] = 1

            # Recompute attention with mask
            attended, attention_weights = self.attention(
                encoded, encoded, encoded, mask
            )

        # Process messages
        messages = self.message_processor(attended)

        # Gate messages
        gated_messages = messages * gates

        # Combine with original encoding
        combined = torch.cat([encoded, gated_messages], dim=-1)
        enhanced_obs = self.output_projection(combined)

        if return_attention:
            # Average attention weights over heads
            attn = attention_weights.mean(dim=1)  # (batch, n_agents, n_agents)
            return enhanced_obs, attn

        return enhanced_obs, None

    def get_communication_cost(self) -> int:
        """
        Get communication cost per agent.

        Returns:
            Number of messages sent per agent per step
        """
        return self.comm_budget


class GatedCommunicationModule(nn.Module):
    """
    Gated communication module that learns when to communicate.

    More efficient than always-on communication.
    """

    def __init__(self, n_agents: int, obs_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Message generator
        self.message_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated communication.

        Args:
            observations: (batch, n_agents, obs_dim)

        Returns:
            enhanced: (batch, n_agents, hidden_dim)
        """
        observations.size(0)

        # Encode
        encoded = self.encoder(observations)  # (batch, n_agents, hidden_dim)

        # Compute pairwise gates
        enhanced_list = []

        for i in range(self.n_agents):
            agent_encoding = encoded[:, i : i + 1, :]  # (batch, 1, hidden_dim)

            # Compute gates with all other agents
            agent_expanded = agent_encoding.expand(-1, self.n_agents, -1)
            pairwise = torch.cat([agent_expanded, encoded], dim=-1)
            gates = self.gate(pairwise)  # (batch, n_agents, 1)

            # Generate messages
            messages = self.message_gen(encoded)  # (batch, n_agents, hidden_dim)

            # Gate messages
            gated = messages * gates
            aggregated = gated.sum(dim=1, keepdim=True)  # (batch, 1, hidden_dim)

            enhanced_list.append(aggregated)

        enhanced = torch.cat(enhanced_list, dim=1)
        return enhanced
