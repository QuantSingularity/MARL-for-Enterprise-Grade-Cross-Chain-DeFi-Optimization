"""
Graph Neural Network Encoder for MARL

Encodes agent observations and global state using GNN layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from typing import Optional


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
        gnn_type: str = "gcn",
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
                self.gnn_layers.append(
                    GATConv(in_dim, hidden_dim, heads=4, concat=False)
                )
            elif gnn_type == "graph":
                self.gnn_layers.append(GraphConv(in_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
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
        n_layers: int = 2,
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
        self.register_buffer(
            "edge_index", torch.tensor(edge_list, dtype=torch.long).t()
        )

        # GNN encoder
        self.gnn = GNNEncoder(
            node_dim=input_dim,
            edge_dim=0,
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
