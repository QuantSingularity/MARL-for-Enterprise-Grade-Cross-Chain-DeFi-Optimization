"""
MARL Agents Package

Implements multi-agent reinforcement learning algorithms:
- QMIX: Value function factorization for cooperative MARL
- MAPPO: Multi-Agent Proximal Policy Optimization
- Baselines: Independent Q-Learning, Random Agent
"""

from .baselines import IndependentQLearning, RandomAgent
from .communication import AttentionCommunicationModule
from .gnn_encoder import GNNEncoder
from .mappo import MAPPOAgent, MAPPOCritic
from .qmix import QMIXAgent, QMIXMixer

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
