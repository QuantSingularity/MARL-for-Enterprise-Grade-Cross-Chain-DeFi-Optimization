"""
Cross-Chain DeFi Environment Simulator

A lightweight multi-chain simulator that models:
- Multiple blockchain networks (chains)
- Cross-chain bridges with latency and failure risk
- DEX liquidity pools with AMM mechanics
- Gas costs and transaction execution
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BridgeStatus(Enum):
    """Bridge operational status."""

    ACTIVE = "active"
    CONGESTED = "congested"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class Chain:
    """Represents a blockchain network."""

    name: str
    chain_id: int
    block_time: float  # seconds
    base_gas_price: float  # gwei
    gas_volatility: float  # std dev of gas price

    def __post_init__(self):
        self.current_gas_price = self.base_gas_price
        self.block_number = 0

    def update_gas_price(self):
        """Update gas price with random walk."""
        self.current_gas_price = max(
            1.0, self.current_gas_price * (1 + np.random.normal(0, self.gas_volatility))
        )

    def mine_block(self):
        """Mine a new block."""
        self.block_number += 1
        self.update_gas_price()


@dataclass
class Bridge:
    """Represents a cross-chain bridge."""

    name: str
    source_chain: str
    target_chain: str
    capacity: float  # max tokens that can be bridged
    latency_mean: float  # mean latency in seconds
    latency_std: float
    failure_rate: float  # probability of failure
    fee_basis_points: int  # fee in basis points (1/10000)

    def __post_init__(self):
        self.current_load = 0.0
        self.status = BridgeStatus.ACTIVE
        self.failure_history = []
        self.total_volume = 0.0

    def get_latency(self) -> float:
        """Sample bridge latency."""
        if self.status == BridgeStatus.FAILED:
            return float("inf")
        return max(0, np.random.normal(self.latency_mean, self.latency_std))

    def attempt_transfer(self, amount: float) -> Tuple[bool, float]:
        """
        Attempt to bridge tokens.

        Returns:
            (success, actual_latency)
        """
        if self.status == BridgeStatus.FAILED:
            return False, 0.0

        if self.current_load + amount > self.capacity:
            self.status = BridgeStatus.CONGESTED
            return False, 0.0

        # Check for failure
        if np.random.random() < self.failure_rate:
            self.failure_history.append(1)
            return False, 0.0

        self.failure_history.append(0)
        self.current_load += amount
        self.total_volume += amount
        latency = self.get_latency()
        return True, latency

    def update_status(self):
        """Update bridge status based on recent failures."""
        if len(self.failure_history) >= 10:
            recent_failures = sum(self.failure_history[-10:]) / 10
            if recent_failures > 0.3:
                self.status = BridgeStatus.FAILED
            elif recent_failures > 0.1:
                self.status = BridgeStatus.CONGESTED
            else:
                self.status = BridgeStatus.ACTIVE


@dataclass
class Pool:
    """Represents a DEX liquidity pool (constant product AMM)."""

    token_a: str
    token_b: str
    reserve_a: float
    reserve_b: float
    fee_tier: int  # basis points

    def get_price(self) -> float:
        """Get current price of token A in terms of token B."""
        if self.reserve_a == 0:
            return 0.0
        return self.reserve_b / self.reserve_a

    def get_output_amount(self, input_amount: float, input_token: str) -> float:
        """
        Calculate output amount for a swap using constant product formula.

        Args:
            input_amount: Amount of input token
            input_token: Which token is being sold ('A' or 'B')

        Returns:
            Output amount
        """
        fee = input_amount * self.fee_tier / 10000
        amount_in_with_fee = input_amount - fee

        if input_token == "A":
            numerator = amount_in_with_fee * self.reserve_b
            denominator = self.reserve_a + amount_in_with_fee
        else:
            numerator = amount_in_with_fee * self.reserve_a
            denominator = self.reserve_b + amount_in_with_fee

        return numerator / denominator if denominator > 0 else 0.0

    def swap(self, input_amount: float, input_token: str) -> Tuple[float, float]:
        """
        Execute a swap.

        Returns:
            (output_amount, price_impact)
        """
        # Capture pre-swap price for price impact calculation
        price_before = self.get_price() if self.reserve_a > 0 else 0.0

        output_amount = self.get_output_amount(input_amount, input_token)

        if input_token == "A":
            self.reserve_a += input_amount
            self.reserve_b -= output_amount
        else:
            self.reserve_b += input_amount
            self.reserve_a -= output_amount

        price_after = self.get_price()

        price_impact = (
            abs(price_after - price_before) / price_before if price_before > 0 else 0.0
        )

        return output_amount, price_impact

    def add_liquidity(self, amount_a: float, amount_b: float):
        """Add liquidity to the pool."""
        self.reserve_a += amount_a
        self.reserve_b += amount_b


class CrossChainEnv(gym.Env):
    """
    Multi-Chain DeFi Environment.

    Models multiple chains connected by bridges with DEX pools on each chain.
    Agents can perform actions like swapping, bridging, and managing liquidity.

    State space includes:
    - Pool reserves on each chain
    - Bridge statuses and loads
    - Gas prices on each chain
    - Current token balances

    Action space includes:
    - Swap tokens on a chain
    - Bridge tokens between chains
    - Add/remove liquidity
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        chains: List[Chain],
        bridges: List[Bridge],
        pools: Dict[str, List[Pool]],  # chain_name -> list of pools
        initial_balances: Dict[str, float] = None,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.chains = {c.name: c for c in chains}
        self.bridges = bridges
        self.pools = pools
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Initialize balances
        self.initial_balances = initial_balances or {"ETH": 100.0, "USDC": 100000.0}
        self.balances = self.initial_balances.copy()

        # Time tracking
        self.current_step = 0
        self.total_time = 0.0

        # Episode history
        self.history = {"swaps": [], "bridges": [], "rewards": [], "gas_spent": 0.0}

        # Define action and observation spaces
        self._setup_action_space()
        self._setup_observation_space()

    def _setup_action_space(self):
        """Define the action space."""
        # Actions: [action_type, chain_idx, amount, target]
        # action_type: 0=swap, 1=bridge, 2=add_liquidity, 3=remove_liquidity, 4=noop
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([4, len(self.chains) - 1, 1000, len(self.chains) - 1]),
            dtype=np.float32,
        )

    def _setup_observation_space(self):
        """Define the observation space."""
        # Observation includes:
        # - Pool reserves (2 per pool)
        # - Bridge statuses (1 per bridge)
        # - Gas prices (1 per chain)
        # - Balances (per token)
        # - Time features

        num_pools = sum(len(pools) for pools in self.pools.values())
        num_bridges = len(self.bridges)
        num_chains = len(self.chains)
        num_tokens = len(self.balances)

        obs_dim = num_pools * 2 + num_bridges + num_chains + num_tokens + 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = []

        # Pool reserves
        for chain_name in sorted(self.chains.keys()):
            for pool in self.pools.get(chain_name, []):
                obs.extend([pool.reserve_a, pool.reserve_b])

        # Bridge statuses
        for bridge in self.bridges:
            status_val = {
                BridgeStatus.ACTIVE: 1.0,
                BridgeStatus.CONGESTED: 0.5,
                BridgeStatus.FAILED: 0.0,
                BridgeStatus.MAINTENANCE: 0.25,
            }[bridge.status]
            obs.append(status_val)

        # Gas prices
        for chain_name in sorted(self.chains.keys()):
            obs.append(self.chains[chain_name].current_gas_price)

        # Balances
        for token in sorted(self.balances.keys()):
            obs.append(self.balances[token])

        # Time features
        obs.append(self.current_step / self.max_steps)
        obs.append(self.total_time)

        return np.array(obs, dtype=np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset chains
        for chain in self.chains.values():
            chain.block_number = 0
            chain.current_gas_price = chain.base_gas_price

        # Reset bridges
        for bridge in self.bridges:
            bridge.current_load = 0.0
            bridge.status = BridgeStatus.ACTIVE
            bridge.failure_history = []

        # Reset balances
        self.balances = self.initial_balances.copy()

        # Reset time
        self.current_step = 0
        self.total_time = 0.0

        # Reset history
        self.history = {"swaps": [], "bridges": [], "rewards": [], "gas_spent": 0.0}

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: [action_type, chain_idx, amount, target]

        Returns:
            observation, reward, terminated, truncated, info
        """
        action_type = int(action[0])
        chain_idx = int(action[1])
        amount = float(action[2])
        target = int(action[3])

        chain_names = sorted(self.chains.keys())
        chain_name = chain_names[min(chain_idx, len(chain_names) - 1)]

        reward = 0.0
        info = {"action_taken": "noop", "success": True}

        # Execute action
        if action_type == 0:  # Swap
            reward, info = self._execute_swap(chain_name, amount)
        elif action_type == 1:  # Bridge
            target_chain = chain_names[min(target, len(chain_names) - 1)]
            reward, info = self._execute_bridge(chain_name, target_chain, amount)
        elif action_type == 2:  # Add liquidity
            reward, info = self._execute_add_liquidity(chain_name, amount)
        elif action_type == 3:  # Remove liquidity
            reward, info = self._execute_remove_liquidity(chain_name, amount)
        else:  # Noop
            reward = -0.01  # Small penalty for inaction
            info = {"action_taken": "noop", "success": True}

        # Update environment state
        self._update_environment()

        # Track history
        self.history["rewards"].append(reward)

        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_observation(), reward, terminated, truncated, info

    def _execute_swap(self, chain_name: str, amount: float) -> Tuple[float, Dict]:
        """Execute a token swap."""
        pools = self.pools.get(chain_name, [])
        if not pools:
            return -1.0, {
                "action_taken": "swap",
                "success": False,
                "reason": "no_pools",
            }

        # Use first pool for simplicity
        pool = pools[0]

        # Check balance
        if self.balances.get(pool.token_a, 0) < amount:
            return -1.0, {
                "action_taken": "swap",
                "success": False,
                "reason": "insufficient_balance",
            }

        # Execute swap
        output, price_impact = pool.swap(amount, "A")

        # Update balances
        self.balances[pool.token_a] -= amount
        self.balances[pool.token_b] = self.balances.get(pool.token_b, 0) + output

        # Pay gas
        gas_cost = self.chains[chain_name].current_gas_price * 21000 / 1e9
        self.history["gas_spent"] += gas_cost

        # Reward: output amount minus gas cost, penalize high price impact
        reward = output / 1000 - gas_cost - price_impact * 10

        self.history["swaps"].append(
            {
                "chain": chain_name,
                "input": amount,
                "output": output,
                "price_impact": price_impact,
            }
        )

        return reward, {
            "action_taken": "swap",
            "success": True,
            "output": output,
            "price_impact": price_impact,
        }

    def _execute_bridge(
        self, source_chain: str, target_chain: str, amount: float
    ) -> Tuple[float, Dict]:
        """Execute a bridge transfer."""
        # Find bridge
        bridge = None
        for b in self.bridges:
            if b.source_chain == source_chain and b.target_chain == target_chain:
                bridge = b
                break

        if bridge is None:
            return -1.0, {
                "action_taken": "bridge",
                "success": False,
                "reason": "no_bridge",
            }

        # Check balance
        if self.balances.get("ETH", 0) < amount:
            return -1.0, {
                "action_taken": "bridge",
                "success": False,
                "reason": "insufficient_balance",
            }

        # Attempt transfer
        success, latency = bridge.attempt_transfer(amount)

        if not success:
            return -2.0, {
                "action_taken": "bridge",
                "success": False,
                "reason": "bridge_failed",
            }

        fee = amount * bridge.fee_basis_points / 10000
        self.balances["ETH"] -= fee

        # Pay gas on source chain
        gas_cost = self.chains[source_chain].current_gas_price * 100000 / 1e9
        self.history["gas_spent"] += gas_cost

        # Reward: negative of fees and latency
        reward = -(fee + gas_cost + latency / 1000)

        self.total_time += latency

        self.history["bridges"].append(
            {
                "source": source_chain,
                "target": target_chain,
                "amount": amount,
                "fee": fee,
                "latency": latency,
            }
        )

        return reward, {
            "action_taken": "bridge",
            "success": True,
            "fee": fee,
            "latency": latency,
        }

    def _execute_add_liquidity(
        self, chain_name: str, amount: float
    ) -> Tuple[float, Dict]:
        """Add liquidity to a pool."""
        pools = self.pools.get(chain_name, [])
        if not pools:
            return -1.0, {
                "action_taken": "add_liquidity",
                "success": False,
                "reason": "no_pools",
            }

        pool = pools[0]

        # Add equal value of both tokens
        pool.add_liquidity(amount, amount * pool.get_price())

        # Pay gas
        gas_cost = self.chains[chain_name].current_gas_price * 150000 / 1e9
        self.history["gas_spent"] += gas_cost

        reward = -gas_cost + 0.1  # Small positive reward for providing liquidity

        return reward, {"action_taken": "add_liquidity", "success": True}

    def _execute_remove_liquidity(
        self, chain_name: str, amount: float
    ) -> Tuple[float, Dict]:
        """Remove liquidity from a pool."""
        # Simplified: just pay gas
        gas_cost = self.chains[chain_name].current_gas_price * 150000 / 1e9
        self.history["gas_spent"] += gas_cost

        reward = -gas_cost

        return reward, {"action_taken": "remove_liquidity", "success": True}

    def _update_environment(self):
        """Update environment state (mine blocks, update bridges)."""
        # Mine blocks on each chain
        for chain in self.chains.values():
            chain.mine_block()

        # Update bridge statuses
        for bridge in self.bridges:
            bridge.update_status()
            # Decay bridge load
            bridge.current_load *= 0.9

    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
            print(f"Total Time: {self.total_time:.2f}s")
            print(f"Balances: {self.balances}")
            print(f"Gas Spent: {self.history['gas_spent']:.4f}")
            print(f"Cumulative Reward: {sum(self.history['rewards']):.4f}")

            print("\n--- Chains ---")
            for name, chain in self.chains.items():
                print(
                    f"{name}: Block {chain.block_number}, Gas {chain.current_gas_price:.2f} gwei"
                )

            print("\n--- Bridges ---")
            for bridge in self.bridges:
                print(
                    f"{bridge.name}: {bridge.status.value}, Load {bridge.current_load:.2f}"
                )

            print("\n--- Pools ---")
            for chain_name, pools in self.pools.items():
                for pool in pools:
                    print(
                        f"{chain_name}/{pool.token_a}-{pool.token_b}: "
                        f"{pool.reserve_a:.2f}/{pool.reserve_b:.2f}, "
                        f"Price {pool.get_price():.6f}"
                    )

    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            "total_steps": self.current_step,
            "total_time": self.total_time,
            "total_gas_spent": self.history["gas_spent"],
            "cumulative_reward": sum(self.history["rewards"]),
            "num_swaps": len(self.history["swaps"]),
            "num_bridges": len(self.history["bridges"]),
            "final_balances": self.balances.copy(),
        }
