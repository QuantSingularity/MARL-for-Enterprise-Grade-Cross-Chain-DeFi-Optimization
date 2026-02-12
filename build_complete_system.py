#!/usr/bin/env python3
"""
Complete System Builder for MARL Cross-Chain DeFi Project
This script systematically builds all missing components and enhancements.
"""

import os
from pathlib import Path


def create_missing_components():
    """Create all missing code components."""

    components = {
        # Communication module (referenced but incomplete)
        "code/src/agents/communication.py": '''"""
Agent Communication Module for MARL

Implements communication protocols between agents including:
- Message passing
- Attention-based communication
- CommNet-style architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class CommNet(nn.Module):
    """
    CommNet: Learning Multiagent Communication with Backpropagation
    
    Implements iterative message passing between agents.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_agents: int,
        n_comm_rounds: int = 2
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
        batch_size = observations.size(0)
        
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


class AttentionComm(nn.Module):
    """
    Attention-based Communication.
    
    Uses multi-head attention for selective message passing.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_agents: int,
        n_heads: int = 4
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        self.n_heads = n_heads
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention-based communication.
        
        Args:
            hidden_states: (batch, n_agents, hidden_dim)
            
        Returns:
            updated_states: (batch, n_agents, hidden_dim)
        """
        # Self-attention
        attn_out, _ = self.attention(hidden_states, hidden_states, hidden_states)
        
        # Residual connection + layer norm
        hidden_states = self.layer_norm(hidden_states + attn_out)
        
        # Feed-forward + residual
        ff_out = self.ff(hidden_states)
        hidden_states = self.layer_norm(hidden_states + ff_out)
        
        return hidden_states


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
        self.messages.append({
            'sender': sender_id,
            'content': message,
            'metadata': metadata or {}
        })
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
    def retrieve(self, agent_id: int, filter_fn=None) -> List[dict]:
        """Retrieve messages for an agent."""
        if filter_fn is None:
            return self.messages
        return [msg for msg in self.messages if filter_fn(msg, agent_id)]
        
    def clear(self):
        """Clear all messages."""
        self.messages = []
''',
        # GNN Encoder (referenced but incomplete)
        "code/src/agents/gnn_encoder.py": '''"""
Graph Neural Network Encoder for MARL

Encodes agent observations and global state using GNN layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from typing import Optional, Tuple

class GNNEncoder(nn.Module):
    """
    GNN-based encoder for multi-agent state representation.
    
    Treats agents as nodes in a graph and learns embeddings.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        gnn_type: str = "gcn"
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.gnn_type = gnn_type
        
        # Input projection
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            
            if gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(in_dim, hidden_dim))
            elif gnn_type == "gat":
                self.gnn_layers.append(GATConv(in_dim, hidden_dim, heads=4, concat=False))
            elif gnn_type == "graph":
                self.gnn_layers.append(GraphConv(in_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
                
        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features (n_nodes, node_dim)
            edge_index: Edge connectivity (2, n_edges)
            edge_attr: Edge features (n_edges, edge_dim)
            
        Returns:
            Node embeddings (n_nodes, output_dim)
        """
        # Encode nodes
        h = F.relu(self.node_encoder(x))
        
        # GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            h_new = gnn_layer(h, edge_index)
            h_new = self.layer_norms[i](h_new)
            h = F.relu(h_new) + h  # Residual connection
            
        # Output
        out = self.output_layer(h)
        
        return out


class FullyConnectedGNN(nn.Module):
    """
    Fully connected GNN for multi-agent systems.
    
    Assumes all agents can communicate (fully connected graph).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_agents: int,
        n_layers: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_agents = n_agents
        self.n_layers = n_layers
        
        # Create fully connected edge index
        edge_list = []
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    edge_list.append([i, j])
        self.register_buffer('edge_index', torch.tensor(edge_list, dtype=torch.long).t())
        
        # GNN encoder
        self.gnn = GNNEncoder(
            node_dim=input_dim,
            edge_dim=0,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            gnn_type="gcn"
        )
        
    def forward(self, agent_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            agent_observations: (batch, n_agents, input_dim)
            
        Returns:
            embeddings: (batch, n_agents, output_dim)
        """
        batch_size = agent_observations.size(0)
        
        # Flatten batch and agents
        x = agent_observations.view(-1, self.input_dim)
        
        # Adjust edge index for batch
        edge_index_list = []
        for b in range(batch_size):
            offset = b * self.n_agents
            edge_index_list.append(self.edge_index + offset)
        edge_index = torch.cat(edge_index_list, dim=1)
        
        # Forward through GNN
        embeddings = self.gnn(x, edge_index)
        
        # Reshape back to (batch, n_agents, output_dim)
        embeddings = embeddings.view(batch_size, self.n_agents, -1)
        
        return embeddings
''',
        # Evaluation metrics module
        "code/src/eval/metrics.py": '''"""
Evaluation Metrics for MARL Agents

Comprehensive metrics for evaluating DeFi optimization performance.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class EpisodeMetrics:
    """Container for episode-level metrics."""
    episode_reward: float
    episode_length: int
    total_gas_spent: float
    num_swaps: int
    num_bridges: int
    avg_price_impact: float
    total_slippage: float
    capital_efficiency: float
    sharpe_ratio: float
    final_portfolio_value: float

class MetricsCalculator:
    """Calculate comprehensive performance metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio of returns."""
        if len(returns) < 2:
            return 0.0
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def calculate_max_drawdown(cumulative_rewards: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_rewards) < 2:
            return 0.0
        cumulative = np.array(cumulative_rewards)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        return float(np.min(drawdown))
    
    @staticmethod
    def calculate_capital_efficiency(
        initial_capital: float,
        final_capital: float,
        locked_capital: float
    ) -> float:
        """Calculate capital efficiency ratio."""
        if locked_capital == 0:
            return 0.0
        profit = final_capital - initial_capital
        return profit / locked_capital
    
    @staticmethod
    def calculate_win_rate(rewards: List[float]) -> float:
        """Calculate percentage of positive reward episodes."""
        if len(rewards) == 0:
            return 0.0
        positive = sum(1 for r in rewards if r > 0)
        return positive / len(rewards)
    
    @staticmethod
    def calculate_volatility(returns: List[float]) -> float:
        """Calculate volatility (std dev) of returns."""
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns))
    
    @staticmethod
    def aggregate_metrics(episode_data_list: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across multiple episodes."""
        rewards = [ep["episode_reward"] for ep in episode_data_list]
        lengths = [ep["episode_length"] for ep in episode_data_list]
        
        cumulative_rewards = np.cumsum(rewards)
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "sharpe_ratio": MetricsCalculator.calculate_sharpe_ratio(rewards),
            "max_drawdown": MetricsCalculator.calculate_max_drawdown(cumulative_rewards.tolist()),
            "win_rate": MetricsCalculator.calculate_win_rate(rewards),
            "volatility": MetricsCalculator.calculate_volatility(rewards),
            "total_episodes": len(episode_data_list)
        }
''',
    }

    # Create all components
    for filepath, content in components.items():
        full_path = Path(filepath)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        print(f"✓ Created: {filepath}")


if __name__ == "__main__":
    os.chdir("/mnt/user-data/outputs/marl-project-enhanced")
    create_missing_components()
    print("\n✓ All missing components created successfully!")
