"""
Demo script for Cross-Chain Environment

Runs a short rollout demonstrating the environment functionality.
"""

from cross_chain_env import Bridge, Chain, CrossChainEnv, Pool


def create_default_env() -> CrossChainEnv:
    """Create a default two-chain environment for demo."""

    # Define chains
    chains = [
        Chain(
            name="Ethereum",
            chain_id=1,
            block_time=12.0,
            base_gas_price=20.0,
            gas_volatility=0.1,
        ),
        Chain(
            name="Arbitrum",
            chain_id=42161,
            block_time=0.25,
            base_gas_price=0.1,
            gas_volatility=0.05,
        ),
    ]

    # Define bridges
    bridges = [
        Bridge(
            name="Arbitrum Bridge",
            source_chain="Ethereum",
            target_chain="Arbitrum",
            capacity=10000.0,
            latency_mean=600.0,
            latency_std=120.0,
            failure_rate=0.01,
            fee_basis_points=10,
        ),
        Bridge(
            name="Arbitrum Exit",
            source_chain="Arbitrum",
            target_chain="Ethereum",
            capacity=5000.0,
            latency_mean=86400.0,  # ~7 days for optimistic rollup
            latency_std=3600.0,
            failure_rate=0.005,
            fee_basis_points=10,
        ),
    ]

    # Define pools
    pools = {
        "Ethereum": [
            Pool(
                token_a="ETH",
                token_b="USDC",
                reserve_a=1000.0,
                reserve_b=2000000.0,
                fee_tier=30,  # 0.3%
            )
        ],
        "Arbitrum": [
            Pool(
                token_a="ETH",
                token_b="USDC",
                reserve_a=500.0,
                reserve_b=1000000.0,
                fee_tier=30,
            )
        ],
    }

    # Create environment
    env = CrossChainEnv(
        chains=chains,
        bridges=bridges,
        pools=pools,
        initial_balances={"ETH": 10.0, "USDC": 20000.0},
        max_steps=50,
        render_mode="human",
    )

    return env


def run_demo(num_steps: int = 20, seed: int = 42):
    """
    Run a demo rollout of the environment.

    Args:
        num_steps: Number of steps to run
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("Cross-Chain DeFi Environment Demo")
    print("=" * 60)

    # Create environment
    env = create_default_env()

    # Reset environment
    obs, info = env.reset(seed=seed)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Run rollout
    print(f"\nRunning {num_steps} steps...")
    print("-" * 60)

    total_reward = 0.0

    for step in range(num_steps):
        # Sample random action
        action = env.action_space.sample()

        # Take step
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Print step info
        action_names = {
            0: "swap",
            1: "bridge",
            2: "add_liq",
            3: "remove_liq",
            4: "noop",
        }
        action_type = int(action[0])
        action_name = action_names.get(action_type, "unknown")

        print(
            f"Step {step+1:2d}: {action_name:12s} | "
            f"Reward: {reward:8.4f} | "
            f"Success: {info.get('success', False)}"
        )

        if terminated or truncated:
            print(f"Episode finished at step {step+1}")
            break

    # Render final state
    print("\n" + "=" * 60)
    print("Final State")
    print("=" * 60)
    env.render()

    # Print statistics
    stats = env.get_stats()
    print("\n" + "=" * 60)
    print("Episode Statistics")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:12.4f}")
        else:
            print(f"{key:25s}: {value}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    return env, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Chain Environment Demo")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_demo(num_steps=args.steps, seed=args.seed)
