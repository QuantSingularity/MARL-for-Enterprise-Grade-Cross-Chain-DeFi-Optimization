"""
Graph Neural Network Encoder for MARL

Encodes agent observations and global state using GNN layers.

This module provides a self-contained GNN implementation that does NOT depend on
torch_geometric (an optional package).  It operates on dense adjacency matrices
and batched node-feature tensors, which is the natural format for MARL where
the number of agents (nodes) is fixed and known at construction time.

Shape convention throughout:
    node_features / x : (batch, n_nodes, feature_dim)
    adj              : (batch, n_nodes, n_nodes)  – can be 0/1 or soft weights
    output           : (batch, n_nodes, output_dim)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Multi-head Graph Attention layer operating on dense adjacency matrices.

    Replaces the torch_geometric GATConv for a dependency-free implementation
    that works directly with the (batch, nodes, features) tensor layout expected
    by the rest of this module and by the test suite.
    """

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads

        assert (
            out_features % n_heads == 0
        ), f"out_features ({out_features}) must be divisible by n_heads ({n_heads})"
        self.head_dim = out_features // n_heads

        # Linear projections for Q, K, V per head
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)

        self.out_proj = nn.Linear(out_features, out_features)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:   (batch, n_nodes, in_features)
            adj: (batch, n_nodes, n_nodes) mask; 0 entries are masked out.
                 If None, full attention (all-ones adjacency) is used.

        Returns:
            out: (batch, n_nodes, out_features)
        """
        batch, n_nodes, _ = x.shape

        Q = self.q_proj(x).view(batch, n_nodes, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(batch, n_nodes, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(batch, n_nodes, self.n_heads, self.head_dim)

        # (batch, n_heads, n_nodes, head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled dot-product attention: (batch, n_heads, n_nodes, n_nodes)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if adj is not None:
            # Expand adj to broadcast over heads: (batch, 1, n_nodes, n_nodes)
            mask = adj.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaNs that appear when an entire row is masked (all -inf → softmax → nan)
        attn = torch.nan_to_num(attn, nan=0.0)

        # (batch, n_heads, n_nodes, head_dim)
        out = torch.matmul(attn, V)

        # Merge heads: (batch, n_nodes, out_features)
        out = (
            out.permute(0, 2, 1, 3).contiguous().view(batch, n_nodes, self.out_features)
        )
        out = self.out_proj(out)
        return out


class GNNEncoder(nn.Module):
    """
    GNN-based encoder for multi-agent state representation.

    Operates on batched (batch, n_nodes, feature_dim) tensors with a dense
    adjacency matrix, making it suitable for MARL without requiring
    torch_geometric.

    Supports 'gcn' (mean-pooled neighbourhood) and 'gat' (attention) layers.
    """

    def __init__(
        self,
        node_dim: Optional[int] = None,
        edge_dim: int = 0,  # kept for API compatibility; not used
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers: int = 2,
        gnn_type: str = "gcn",
        # Alias accepted for backward compatibility with test suite
        node_feature_dim: Optional[int] = None,
    ):
        super().__init__()

        if node_dim is None and node_feature_dim is None:
            raise ValueError("Provide either `node_dim` or `node_feature_dim`.")
        effective_node_dim = node_dim if node_dim is not None else node_feature_dim

        self.node_dim = effective_node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.gnn_type = gnn_type

        # Input projection
        self.node_encoder = nn.Linear(effective_node_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            if gnn_type == "gat":
                self.gnn_layers.append(
                    GraphAttentionLayer(hidden_dim, hidden_dim, n_heads=4)
                )
            elif gnn_type in ("gcn", "graph"):
                # Simple GCN: linear applied after mean-aggregation of neighbours
                self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                raise ValueError(
                    f"Unknown GNN type: {gnn_type}. Choose 'gcn', 'gat', or 'graph'."
                )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

    def _gcn_conv(
        self, h: torch.Tensor, adj: Optional[torch.Tensor], layer: nn.Linear
    ) -> torch.Tensor:
        """
        Simple GCN convolution: aggregate neighbour features then apply linear.

        Args:
            h:   (batch, n_nodes, hidden_dim)
            adj: (batch, n_nodes, n_nodes) or None (full graph)
            layer: linear layer to apply after aggregation

        Returns:
            (batch, n_nodes, hidden_dim)
        """
        if adj is not None:
            # Row-normalise adjacency (add self-loops implicitly via identity)
            degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
            norm_adj = adj / degree  # (batch, n_nodes, n_nodes)
            agg = torch.bmm(norm_adj, h)  # (batch, n_nodes, hidden_dim)
        else:
            agg = h  # No adjacency → identity (self only)
        return layer(agg)

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        # Legacy positional arg: edge_index — ignored (not used in this impl)
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GNN.

        Args:
            x:   (batch, n_nodes, node_dim)  or  (n_nodes, node_dim) for single graph
            adj: (batch, n_nodes, n_nodes)   or  (n_nodes, n_nodes)   — optional

        Returns:
            Node embeddings: same leading batch dims, last dim = output_dim
        """
        # Handle un-batched input for backward compatibility
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)
            if adj is not None:
                adj = adj.unsqueeze(0)

        # Encode nodes
        h = F.relu(self.node_encoder(x))

        # GNN layers with residual connections and layer norm
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == "gat":
                h_new = gnn_layer(h, adj)
            else:
                h_new = self._gcn_conv(h, adj, gnn_layer)

            h = F.relu(self.layer_norms[i](h_new)) + h

        # Output projection
        out = self.output_layer(h)

        if unbatched:
            out = out.squeeze(0)

        return out

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        Compute a graph-level embedding by pooling node embeddings.

        Args:
            x:    (batch, n_nodes, node_dim)
            adj:  (batch, n_nodes, n_nodes) — optional
            pool: 'mean' | 'max' | 'sum'

        Returns:
            graph_emb: (batch, output_dim)
        """
        node_emb = self.forward(x, adj)  # (batch, n_nodes, output_dim)

        if pool == "mean":
            return node_emb.mean(dim=1)
        elif pool == "max":
            return node_emb.max(dim=1).values
        elif pool == "sum":
            return node_emb.sum(dim=1)
        else:
            raise ValueError(
                f"Unknown pooling: {pool}. Choose 'mean', 'max', or 'sum'."
            )


class FullyConnectedGNN(nn.Module):
    """
    Fully connected GNN for multi-agent systems.

    Assumes all agents can communicate (fully connected graph).
    Uses a dense adjacency matrix (all ones, no self-loops) so that every
    agent attends to every other agent.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_agents: int,
        n_layers: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_agents = n_agents
        self.n_layers = n_layers

        # Fully connected adjacency (all-ones, including self-connections)
        adj = torch.ones(n_agents, n_agents)
        self.register_buffer("adj", adj)

        # GNN encoder (uses our dependency-free implementation)
        self.gnn = GNNEncoder(
            node_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            gnn_type="gcn",
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

        # Expand adjacency to batch dimension
        adj_batch = self.adj.unsqueeze(0).expand(batch_size, -1, -1)

        # Forward through GNN
        embeddings = self.gnn(agent_observations, adj_batch)

        return embeddings
