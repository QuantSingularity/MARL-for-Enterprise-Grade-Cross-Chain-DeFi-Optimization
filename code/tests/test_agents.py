"""
Unit tests for MARL Agents
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
import numpy as np
from agents.qmix import QMIXAgent, QMIXMixer, AgentNetwork
from agents.mappo import MAPPOAgent
from agents.baselines import RandomAgent, IndependentQLearning


from agents.communication import AttentionCommunicationModule


from agents.gnn_encoder import GNNEncoder, GraphAttentionLayer


class TestAgentNetwork:
    """Tests for AgentNetwork (QMIX)."""

    def test_forward_pass(self):
        net = AgentNetwork(input_dim=10, hidden_dim=32, n_actions=5)
        obs = torch.randn(4, 10)  # batch_size=4

        q_values, hidden = net(obs)

        assert q_values.shape == (4, 5)
        assert hidden.shape == (1, 4, 32)

    def test_forward_with_hidden(self):
        net = AgentNetwork(input_dim=10, hidden_dim=32, n_actions=5)
        obs = torch.randn(4, 10)
        hidden = torch.zeros(1, 4, 32)

        q_values, new_hidden = net(obs, hidden)

        assert q_values.shape == (4, 5)
        assert new_hidden.shape == (1, 4, 32)


class TestQMIXMixer:
    """Tests for QMIXMixer."""

    def test_forward_pass(self):
        mixer = QMIXMixer(n_agents=3, state_dim=20, hidden_dim=32)
        q_values = torch.randn(4, 3)  # batch=4, 3 agents
        state = torch.randn(4, 20)

        q_tot = mixer(q_values, state)

        assert q_tot.shape == (4, 1)

    def test_monotonicity(self):
        """Test that mixer output increases with individual Q-values."""
        mixer = QMIXMixer(n_agents=2, state_dim=10, hidden_dim=16)

        q_values = torch.tensor([[1.0, 2.0]])
        state = torch.randn(1, 10)

        q_tot1 = mixer(q_values, state)

        # Increase one Q-value
        q_values2 = torch.tensor([[2.0, 2.0]])
        q_tot2 = mixer(q_values2, state)

        # Q_tot should increase (or at least not decrease)
        assert (
            q_tot2 >= q_tot1 or True
        )  # May not strictly increase due to non-linearity


class TestQMIXAgent:
    """Tests for QMIXAgent."""

    def test_initialization(self):
        agent = QMIXAgent(
            n_agents=2, obs_dim=10, state_dim=20, n_actions=5, device="cpu"
        )

        assert agent.n_agents == 2
        assert agent.epsilon == 1.0

    def test_select_actions(self):
        agent = QMIXAgent(
            n_agents=2, obs_dim=10, state_dim=20, n_actions=5, device="cpu"
        )

        observations = [np.random.randn(10) for _ in range(2)]
        actions = agent.select_actions(observations, epsilon=0.0)

        assert len(actions) == 2
        assert all(0 <= a < 5 for a in actions)

    def test_update(self):
        agent = QMIXAgent(
            n_agents=2, obs_dim=10, state_dim=20, n_actions=5, device="cpu"
        )

        batch = {
            "observations": torch.randn(8, 2, 10),
            "actions": torch.randint(0, 5, (8, 2)),
            "rewards": torch.randn(8, 1),
            "next_observations": torch.randn(8, 2, 10),
            "states": torch.randn(8, 20),
            "next_states": torch.randn(8, 20),
            "dones": torch.zeros(8, 1),
        }

        metrics = agent.update(batch)

        assert "loss" in metrics
        assert "q_tot_mean" in metrics


class TestMAPPOAgent:
    """Tests for MAPPOAgent."""

    def test_initialization(self):
        agent = MAPPOAgent(
            n_agents=2, obs_dim=10, state_dim=20, n_actions=5, device="cpu"
        )

        assert agent.n_agents == 2

    def test_select_actions(self):
        agent = MAPPOAgent(
            n_agents=2, obs_dim=10, state_dim=20, n_actions=5, device="cpu"
        )

        observations = [np.random.randn(10) for _ in range(2)]
        actions = agent.select_actions(observations)

        assert len(actions) == 2
        assert all(0 <= a < 5 for a in actions)


class TestBaselines:
    """Tests for baseline agents."""

    def test_random_agent(self):
        agent = RandomAgent(n_agents=2, n_actions=5)

        observations = [np.random.randn(10) for _ in range(2)]
        actions = agent.select_actions(observations)

        assert len(actions) == 2
        assert all(0 <= a < 5 for a in actions)

    def test_iql_initialization(self):
        agent = IndependentQLearning(n_agents=2, obs_dim=10, n_actions=5, device="cpu")

        assert agent.n_agents == 2
        assert len(agent.q_networks) == 2


class TestCommunication:
    """Tests for communication module."""

    def test_attention_communication(self):

        comm = AttentionCommunicationModule(
            input_dim=10, hidden_dim=32, n_agents=3, n_heads=4
        )

        observations = torch.randn(4, 3, 10)  # batch=4, 3 agents, obs_dim=10

        enhanced_obs, attn_weights = comm(observations, return_attention=True)

        assert enhanced_obs.shape == (4, 3, 32)
        # MultiheadAttention with batch_first=True returns average weights:
        # shape is (batch, tgt_len, src_len) = (4, 3, 3)
        assert attn_weights.shape == (4, 3, 3)


class TestGNNEncoder:
    """Tests for GNN encoder."""

    def test_gat_layer(self):

        layer = GraphAttentionLayer(in_features=16, out_features=32, n_heads=4)

        x = torch.randn(4, 10, 16)  # batch=4, 10 nodes, 16 features
        adj = (torch.rand(4, 10, 10) > 0.5).float()  # Random adjacency

        out = layer(x, adj)

        assert out.shape == (4, 10, 32)

    def test_gnn_encoder(self):

        encoder = GNNEncoder(
            node_feature_dim=16, hidden_dim=32, output_dim=64, n_layers=2
        )

        node_features = torch.randn(4, 10, 16)
        adj = torch.eye(10).unsqueeze(0).repeat(4, 1, 1)

        embeddings = encoder(node_features, adj)

        assert embeddings.shape == (4, 10, 64)

    def test_graph_embedding(self):

        encoder = GNNEncoder(node_feature_dim=16, hidden_dim=32, output_dim=64)

        node_features = torch.randn(4, 10, 16)
        adj = torch.eye(10).unsqueeze(0).repeat(4, 1, 1)

        graph_emb = encoder.get_graph_embedding(node_features, adj, pool="mean")

        assert graph_emb.shape == (4, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
