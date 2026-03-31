"""
Multi-Chain DeFi Environment Package

This package provides a lightweight simulator for cross-chain DeFi operations,
including DEX swaps, bridge transfers, and liquidity management.
"""

from .cross_chain_env import Bridge, Chain, CrossChainEnv, Pool
from .demo_env import run_demo

__all__ = ["CrossChainEnv", "Chain", "Bridge", "Pool", "run_demo"]
