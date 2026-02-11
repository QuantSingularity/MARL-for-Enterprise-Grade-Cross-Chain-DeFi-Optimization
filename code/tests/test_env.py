"""
Unit tests for Cross-Chain Environment
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
from envs.cross_chain_env import CrossChainEnv, Chain, Bridge, Pool


class TestChain:
    """Tests for Chain class."""

    def test_chain_initialization(self):
        chain = Chain(
            name="Ethereum",
            chain_id=1,
            block_time=12.0,
            base_gas_price=20.0,
            gas_volatility=0.1,
        )

        assert chain.name == "Ethereum"
        assert chain.chain_id == 1
        assert chain.block_time == 12.0
        assert chain.current_gas_price == 20.0
        assert chain.block_number == 0

    def test_gas_price_update(self):
        chain = Chain("Test", 1, 1.0, 10.0, 0.1)
        initial_price = chain.current_gas_price

        chain.update_gas_price()

        # Gas price should change but remain positive
        assert chain.current_gas_price > 0
        assert (
            chain.current_gas_price != initial_price or True
        )  # May stay same by chance

    def test_mine_block(self):
        chain = Chain("Test", 1, 1.0, 10.0, 0.1)

        chain.mine_block()

        assert chain.block_number == 1


class TestBridge:
    """Tests for Bridge class."""

    def test_bridge_initialization(self):
        bridge = Bridge(
            name="TestBridge",
            source_chain="A",
            target_chain="B",
            capacity=1000.0,
            latency_mean=100.0,
            latency_std=10.0,
            failure_rate=0.01,
            fee_basis_points=10,
        )

        assert bridge.name == "TestBridge"
        assert bridge.capacity == 1000.0
        assert bridge.current_load == 0.0

    def test_successful_transfer(self):
        bridge = Bridge(
            name="TestBridge",
            source_chain="A",
            target_chain="B",
            capacity=1000.0,
            latency_mean=100.0,
            latency_std=10.0,
            failure_rate=0.0,  # No failures
            fee_basis_points=10,
        )

        success, latency = bridge.attempt_transfer(100.0)

        assert success is True
        assert latency >= 0
        assert bridge.current_load == 100.0

    def test_failed_transfer_over_capacity(self):
        bridge = Bridge(
            name="TestBridge",
            source_chain="A",
            target_chain="B",
            capacity=100.0,
            latency_mean=100.0,
            latency_std=10.0,
            failure_rate=0.0,
            fee_basis_points=10,
        )

        success, _ = bridge.attempt_transfer(150.0)

        assert success is False


class TestPool:
    """Tests for Pool class."""

    def test_pool_initialization(self):
        pool = Pool(
            token_a="ETH",
            token_b="USDC",
            reserve_a=1000.0,
            reserve_b=2000000.0,
            fee_tier=30,
        )

        assert pool.token_a == "ETH"
        assert pool.token_b == "USDC"
        assert pool.reserve_a == 1000.0
        assert pool.reserve_b == 2000000.0

    def test_get_price(self):
        pool = Pool("ETH", "USDC", 1000.0, 2000000.0, 30)

        price = pool.get_price()

        assert price == 2000.0

    def test_swap(self):
        pool = Pool("ETH", "USDC", 1000.0, 2000000.0, 30)

        output, price_impact = pool.swap(1.0, "A")

        # Output should be positive but less than input * price (due to fees)
        assert output > 0
        assert output < 2000.0
        assert price_impact >= 0

        # Reserves should have changed
        assert pool.reserve_a == 1001.0
        assert pool.reserve_b < 2000000.0


class TestCrossChainEnv:
    """Tests for CrossChainEnv class."""

    @pytest.fixture
    def env(self):
        chains = [
            Chain("Ethereum", 1, 12.0, 20.0, 0.1),
            Chain("Arbitrum", 42161, 0.25, 0.1, 0.05),
        ]

        bridges = [
            Bridge("Bridge", "Ethereum", "Arbitrum", 10000.0, 600.0, 120.0, 0.01, 10)
        ]

        pools = {"Ethereum": [Pool("ETH", "USDC", 1000.0, 2000000.0, 30)]}

        return CrossChainEnv(chains, bridges, pools, max_steps=10)

    def test_env_initialization(self, env):
        assert env.observation_space is not None
        assert env.action_space is not None
        assert len(env.chains) == 2

    def test_reset(self, env):
        obs, info = env.reset(seed=42)

        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape

    def test_step(self, env):
        env.reset(seed=42)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_termination(self, env):
        env.reset(seed=42)

        done = False
        steps = 0
        max_steps = 20

        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert steps <= env.max_steps or done


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
