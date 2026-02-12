"""Integration tests for complete training pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.cross_chain_env import CrossChainEnv, Chain, Pool
from agents.qmix import QMIXAgent
from agents.mappo import MAPPOAgent


def create_test_env():
    """Create a minimal test environment."""
    chains = [Chain("TestChain", 1, 1.0, 10.0, 0.1)]
    bridges = []
    pools = {"TestChain": [Pool("ETH", "USDC", 100.0, 200000.0, 30)]}
    return CrossChainEnv(chains, bridges, pools, max_steps=10)


def test_qmix_training():
    """Test QMIX agent training."""
    env = create_test_env()
    agent = QMIXAgent(n_agents=2, obs_dim=10, state_dim=20, n_actions=5, hidden_dim=32)
    obs, _ = env.reset()
    agent_obs = [obs[:10], obs[10:20]]
    actions = agent.select_actions(agent_obs)
    assert len(actions) == 2
    assert all(0 <= a < 5 for a in actions)


def test_mappo_training():
    """Test MAPPO agent training."""
    env = create_test_env()
    agent = MAPPOAgent(n_agents=2, obs_dim=10, state_dim=20, n_actions=5, hidden_dim=32)
    obs, _ = env.reset()
    agent_obs = [obs[:10], obs[10:20]]
    actions = agent.select_actions(agent_obs)
    assert len(actions) == 2
    assert all(0 <= a < 5 for a in actions)
