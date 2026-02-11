"""
MARL Agents Package

Implements multi-agent reinforcement learning algorithms:
- QMIX: Value function factorization for cooperative MARL
- MAPPO: Multi-Agent Proximal Policy Optimization
- Baselines: Independent Q-Learning, Random Agent
"""

from .qmix import QMIXAgent, QMIXMixer
from .mappo import MAPPOAgent, MAPPOCritic
from .baselines import RandomAgent, IndependentQLearning
from .communication import AttentionCommunicationModule
from .gnn_encoder import GNNEncoder

__all__ = [
    "QMIXAgent",
    "QMIXMixer",
    "MAPPOAgent",
    "MAPPOCritic",
    "RandomAgent",
    "IndependentQLearning",
    "AttentionCommunicationModule",
    "GNNEncoder",
]
