"""
Graph Neural Network Encoder for Cross-Chain State Representation

Implements Graph Attention Networks (GAT) for encoding the cross-chain
DeFi graph structure (chains, bridges, pools as nodes/edges).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer (GAT).

    Based on: Velickovic et al. "Graph Attention Networks" (ICLR 2018)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.d_k = out_features // n_heads

        # Linear transformation for each head
        self.W = nn.Parameter(torch.zeros(n_heads, in_features, self.d_k))
        nn.init.xavier_uniform_(self.W)

        # Attention parameters
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.d_k, 1))
        nn.init.xavier_uniform_(self.a)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (batch, n_nodes, in_features)
            adj: Adjacency matrix (batch, n_nodes, n_nodes)

        Returns:
            Updated node features (batch, n_nodes, out_features)
        """
        batch_size, n_nodes, _ = x.size()

        # Transform features for each head
        h = torch.einsum("bni,hio->bhno", x, self.W)  # (batch, heads, nodes, d_k)

        # Compute attention scores
        # Repeat for source and target
        h_i = h.unsqueeze(3).expand(
            -1, -1, -1, n_nodes, -1
        )  # (batch, heads, nodes, nodes, d_k)
        h_j = h.unsqueeze(2).expand(
            -1, -1, n_nodes, -1, -1
        )  # (batch, heads, nodes, nodes, d_k)

        # Concatenate and compute attention
        a_input = torch.cat([h_i, h_j], dim=-1)  # (batch, heads, nodes, nodes, 2*d_k)
        e = torch.einsum("bhnno,hoa->bhnn", a_input, self.a).squeeze(
            -1
        )  # (batch, heads, nodes, nodes)

        e = self.leakyrelu(e)

        # Mask with adjacency matrix
        mask = adj.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        e = e.masked_fill(mask == 0, -1e9)

        # Softmax
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)

        # Apply attention
        h_prime = torch.einsum(
            "bhnn,bhni->bhni", alpha, h
        )  # (batch, heads, nodes, d_k)

        if self.concat:
            # Concatenate heads
            out = h_prime.view(batch_size, n_nodes, -1)
        else:
            # Average heads
            out = h_prime.mean(dim=1)

        return out


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for cross-chain state.

    Encodes the cross-chain DeFi graph (chains, bridges, pools)
    into node embeddings for agent decision-making.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize GNN encoder.

        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            n_layers: Number of GAT layers
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        # Input projection
        self.input_projection = nn.Linear(node_feature_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            out_dim = hidden_dim if i < n_layers - 1 else output_dim

            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=(i < n_layers - 1),  # Don't concat on last layer
                )
            )

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm(hidden_dim if i < n_layers - 1 else output_dim)
                for i in range(n_layers)
            ]
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: (batch, n_nodes, node_feature_dim)
            adjacency_matrix: (batch, n_nodes, n_nodes)

        Returns:
            node_embeddings: (batch, n_nodes, output_dim)
        """
        x = self.input_projection(node_features)
        x = self.dropout(x)

        # Apply GAT layers
        for i, (gat_layer, layer_norm) in enumerate(
            zip(self.gat_layers, self.layer_norms)
        ):
            x_new = gat_layer(x, adjacency_matrix)
            x_new = layer_norm(x_new)

            if i < self.n_layers - 1:
                x_new = F.elu(x_new)
                x_new = self.dropout(x_new)
                # Residual connection
                x = x + x_new if x.size() == x_new.size() else x_new
            else:
                x = x_new

        return x

    def get_graph_embedding(
        self,
        node_features: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        Get a single graph-level embedding.

        Args:
            node_features: (batch, n_nodes, node_feature_dim)
            adjacency_matrix: (batch, n_nodes, n_nodes)
            pool: Pooling method ('mean', 'sum', 'max')

        Returns:
            graph_embedding: (batch, output_dim)
        """
        node_embeddings = self.forward(node_features, adjacency_matrix)

        if pool == "mean":
            return node_embeddings.mean(dim=1)
        elif pool == "sum":
            return node_embeddings.sum(dim=1)
        elif pool == "max":
            return node_embeddings.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {pool}")


class CrossChainGraphBuilder:
    """
    Builds graph representations of cross-chain DeFi state.

    Converts environment state into node features and adjacency matrices.
    """

    def __init__(self, n_chains: int, n_bridges: int, n_pools_per_chain: int):
        self.n_chains = n_chains
        n_pools = n_chains * n_pools_per_chain
        self.n_nodes = n_chains + n_bridges + n_pools

        # Node type indices
        self.chain_indices = list(range(n_chains))
        self.bridge_indices = list(range(n_chains, n_chains + n_bridges))
        self.pool_indices = list(range(n_chains + n_bridges, self.n_nodes))

    def build_graph(
        self,
        chain_states: torch.Tensor,
        bridge_states: torch.Tensor,
        pool_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from environment state.

        Args:
            chain_states: (batch, n_chains, chain_feature_dim)
            bridge_states: (batch, n_bridges, bridge_feature_dim)
            pool_states: (batch, n_pools, pool_feature_dim)

        Returns:
            node_features: (batch, n_nodes, node_feature_dim)
            adjacency_matrix: (batch, n_nodes, n_nodes)
        """
        batch_size = chain_states.size(0)

        # Pad features to common dimension
        max_dim = max(
            chain_states.size(-1), bridge_states.size(-1), pool_states.size(-1)
        )

        chain_padded = F.pad(chain_states, (0, max_dim - chain_states.size(-1)))
        bridge_padded = F.pad(bridge_states, (0, max_dim - bridge_states.size(-1)))
        pool_padded = F.pad(pool_states, (0, max_dim - pool_states.size(-1)))

        # Concatenate all nodes
        node_features = torch.cat([chain_padded, bridge_padded, pool_padded], dim=1)

        # Build adjacency matrix
        adj = torch.zeros(batch_size, self.n_nodes, self.n_nodes)

        # Chains connected to their pools
        for i, chain_idx in enumerate(self.chain_indices):
            pool_start = self.pool_indices[i]
            pool_end = pool_start + (len(self.pool_indices) // self.n_chains)
            for pool_idx in range(pool_start, pool_end):
                adj[:, chain_idx, pool_idx] = 1
                adj[:, pool_idx, chain_idx] = 1

        # Bridges connect chains
        for i, bridge_idx in enumerate(self.bridge_indices):
            # Connect bridge to source and target chains
            source_chain = i % self.n_chains
            target_chain = (i + 1) % self.n_chains
            adj[:, bridge_idx, self.chain_indices[source_chain]] = 1
            adj[:, self.chain_indices[source_chain], bridge_idx] = 1
            adj[:, bridge_idx, self.chain_indices[target_chain]] = 1
            adj[:, self.chain_indices[target_chain], bridge_idx] = 1

        return node_features, adj


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder for comparison with GNN.

    Flattens the graph structure and processes with MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, input_dim) flattened input

        Returns:
            output: (batch, output_dim)
        """
        return self.network(x)
