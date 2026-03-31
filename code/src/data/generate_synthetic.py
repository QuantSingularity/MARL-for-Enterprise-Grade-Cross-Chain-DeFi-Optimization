"""
Synthetic Data Generator for Cross-Chain DeFi

Generates synthetic datasets with statistical properties similar to
real blockchain data for training and evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PriceModel:
    """Geometric Brownian Motion with jumps for price simulation."""

    def __init__(
        self,
        initial_price: float = 2000.0,
        drift: float = 0.0,
        volatility: float = 0.02,
        jump_intensity: float = 0.01,
        jump_mean: float = 0.0,
        jump_std: float = 0.05,
    ):
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def simulate(
        self, n_steps: int, dt: float = 1.0, seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate price path."""
        if seed is not None:
            np.random.seed(seed)

        prices = np.zeros(n_steps)
        prices[0] = self.initial_price

        for t in range(1, n_steps):
            # GBM component
            dW = np.random.normal(0, np.sqrt(dt))
            dS = self.drift * prices[t - 1] * dt + self.volatility * prices[t - 1] * dW

            # Jump component
            if np.random.random() < self.jump_intensity:
                jump = np.random.normal(self.jump_mean, self.jump_std)
                dS += prices[t - 1] * jump

            prices[t] = max(0.01, prices[t - 1] + dS)

        return prices


class GasPriceModel:
    """Autoregressive model for gas price simulation."""

    def __init__(
        self,
        base_price: float = 20.0,
        ar_coeff: float = 0.8,
        volatility: float = 5.0,
        spike_prob: float = 0.05,
        spike_multiplier: float = 3.0,
    ):
        self.base_price = base_price
        self.ar_coeff = ar_coeff
        self.volatility = volatility
        self.spike_prob = spike_prob
        self.spike_multiplier = spike_multiplier

    def simulate(self, n_steps: int, seed: Optional[int] = None) -> np.ndarray:
        """Simulate gas price path."""
        if seed is not None:
            np.random.seed(seed)

        gas_prices = np.zeros(n_steps)
        gas_prices[0] = self.base_price

        for t in range(1, n_steps):
            # AR(1) component
            noise = np.random.normal(0, self.volatility)
            gas_prices[t] = (
                self.base_price * (1 - self.ar_coeff)
                + self.ar_coeff * gas_prices[t - 1]
                + noise
            )

            # Occasional spikes
            if np.random.random() < self.spike_prob:
                gas_prices[t] *= self.spike_multiplier

            gas_prices[t] = max(1.0, gas_prices[t])

        return gas_prices


class BridgeLatencyModel:
    """Lognormal model for bridge latency."""

    def __init__(
        self,
        mean_latency: float = 600.0,
        std_latency: float = 120.0,
        failure_rate: float = 0.01,
    ):
        self.mean_latency = mean_latency
        self.std_latency = std_latency
        self.failure_rate = failure_rate

        # Convert to lognormal parameters
        self.mu = np.log(mean_latency**2 / np.sqrt(std_latency**2 + mean_latency**2))
        self.sigma = np.sqrt(np.log(1 + std_latency**2 / mean_latency**2))

    def simulate(
        self, n_steps: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate bridge latencies and failures.

        Returns:
            latencies: Array of latencies (inf for failures)
            success: Boolean array indicating success
        """
        if seed is not None:
            np.random.seed(seed)

        latencies = np.random.lognormal(self.mu, self.sigma, n_steps)
        success = np.random.random(n_steps) > self.failure_rate

        latencies[~success] = np.inf

        return latencies, success


class SyntheticDataGenerator:
    """Main class for generating synthetic cross-chain DeFi data."""

    def __init__(
        self,
        n_chains: int = 2,
        n_bridges: int = 2,
        n_pools_per_chain: int = 1,
        tokens: List[str] = None,
    ):
        self.n_chains = n_chains
        self.n_bridges = n_bridges
        self.n_pools_per_chain = n_pools_per_chain
        self.tokens = tokens or ["ETH", "USDC"]

        # Initialize models
        self.price_model = PriceModel()
        self.gas_model = GasPriceModel()
        self.latency_model = BridgeLatencyModel()

    def generate_chain_data(
        self, n_steps: int, seed: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Generate data for each chain."""
        if seed is not None:
            np.random.seed(seed)

        chain_data = {}

        for i in range(self.n_chains):
            # Gas prices
            gas_prices = self.gas_model.simulate(
                n_steps, seed=seed + i if seed else None
            )

            # Block numbers
            block_numbers = np.arange(n_steps)

            # Token prices
            token_prices = {}
            for token in self.tokens:
                if token == "ETH":
                    prices = self.price_model.simulate(
                        n_steps, seed=seed + 100 + i if seed else None
                    )
                else:  # Stablecoins
                    prices = np.ones(n_steps) + np.random.normal(0, 0.001, n_steps)
                token_prices[f"{token}_price"] = prices

            # Pool reserves
            pool_data = {}
            for j in range(self.n_pools_per_chain):
                base_reserve = 1000.0 * (1 + i)  # Different liquidity per chain
                pool_data[f"pool_{j}_reserve_A"] = (
                    base_reserve + np.random.normal(0, 50, n_steps).cumsum()
                )
                pool_data[f"pool_{j}_reserve_B"] = (
                    base_reserve * 2000 + np.random.normal(0, 10000, n_steps).cumsum()
                )

            # Combine
            df = pd.DataFrame(
                {
                    "block_number": block_numbers,
                    "gas_price_gwei": gas_prices,
                    **token_prices,
                    **pool_data,
                }
            )

            chain_data[f"chain_{i}"] = df

        return chain_data

    def generate_bridge_data(
        self, n_steps: int, seed: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """Generate data for each bridge."""
        if seed is not None:
            np.random.seed(seed)

        bridge_data = {}

        for i in range(self.n_bridges):
            latencies, success = self.latency_model.simulate(
                n_steps, seed=seed + i if seed else None
            )

            # Bridge TVL
            tvl = 10000.0 + np.random.normal(0, 500, n_steps).cumsum()

            # Volume
            volume = np.random.exponential(100, n_steps)

            df = pd.DataFrame(
                {
                    "latency_seconds": latencies,
                    "success": success,
                    "tvl": tvl,
                    "volume": volume,
                }
            )

            bridge_data[f"bridge_{i}"] = df

        return bridge_data

    def generate_swap_events(
        self, n_events: int = 1000, seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate synthetic swap events."""
        if seed is not None:
            np.random.seed(seed)

        events = []

        for i in range(n_events):
            event = {
                "timestamp": np.random.randint(0, 1000000),
                "chain_id": np.random.randint(0, self.n_chains),
                "pool_id": np.random.randint(0, self.n_pools_per_chain),
                "token_in": np.random.choice(self.tokens),
                "token_out": np.random.choice(self.tokens),
                "amount_in": np.random.lognormal(0, 1),
                "amount_out": np.random.lognormal(0, 1),
                "gas_used": np.random.normal(150000, 20000),
                "gas_price": np.random.lognormal(3, 0.5),
                "slippage": np.random.exponential(0.001),
            }
            events.append(event)

        return pd.DataFrame(events)

    def generate_all(
        self,
        n_steps: int = 10000,
        n_swap_events: int = 1000,
        seed: int = 42,
        output_dir: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate complete synthetic dataset.

        Args:
            n_steps: Number of time steps
            n_swap_events: Number of swap events
            seed: Random seed
            output_dir: Directory to save data (optional)

        Returns:
            Dictionary of DataFrames
        """
        print(f"Generating synthetic data with seed {seed}...")

        # Generate chain data
        chain_data = self.generate_chain_data(n_steps, seed)

        # Generate bridge data
        bridge_data = self.generate_bridge_data(n_steps, seed + 1000)

        # Generate swap events
        swap_events = self.generate_swap_events(n_swap_events, seed + 2000)

        # Combine
        all_data = {**chain_data, **bridge_data, "swap_events": swap_events}

        # Save if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            for name, df in all_data.items():
                filepath = output_path / f"{name}.csv"
                df.to_csv(filepath, index=False)
                print(f"Saved: {filepath}")

            # Save metadata
            metadata = {
                "n_chains": self.n_chains,
                "n_bridges": self.n_bridges,
                "n_pools_per_chain": self.n_pools_per_chain,
                "tokens": self.tokens,
                "n_steps": n_steps,
                "n_swap_events": n_swap_events,
                "seed": seed,
            }

            with open(output_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Metadata saved: {output_path / 'metadata.json'}")

        print("Data generation complete!")

        return all_data


def main():
    """CLI for synthetic data generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic cross-chain DeFi data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--n-steps", type=int, default=1000, help="Number of time steps"
    )
    parser.add_argument(
        "--n-swaps", type=int, default=500, help="Number of swap events"
    )
    parser.add_argument("--n-chains", type=int, default=2, help="Number of chains")
    parser.add_argument("--n-bridges", type=int, default=2, help="Number of bridges")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generator = SyntheticDataGenerator(n_chains=args.n_chains, n_bridges=args.n_bridges)

    generator.generate_all(
        n_steps=args.n_steps,
        n_swap_events=args.n_swaps,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
