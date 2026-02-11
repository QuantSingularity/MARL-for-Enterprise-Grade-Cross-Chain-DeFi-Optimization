"""
Evaluation Script for Trained MARL Agents

Loads trained checkpoints and evaluates on synthetic environment,
producing results CSV and visualization plots.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json
from tqdm import tqdm

from envs.cross_chain_env import CrossChainEnv, Chain, Bridge, Pool
from agents.qmix import QMIXAgent
from agents.mappo import MAPPOAgent
from agents.baselines import RandomAgent


def create_test_env(seed: int = 123) -> CrossChainEnv:
    """Create a test environment with fixed seed."""

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
            latency_mean=86400.0,
            latency_std=3600.0,
            failure_rate=0.005,
            fee_basis_points=10,
        ),
    ]

    pools = {
        "Ethereum": [
            Pool(
                token_a="ETH",
                token_b="USDC",
                reserve_a=1000.0,
                reserve_b=2000000.0,
                fee_tier=30,
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

    env = CrossChainEnv(
        chains=chains,
        bridges=bridges,
        pools=pools,
        initial_balances={"ETH": 10.0, "USDC": 20000.0},
        max_steps=100,
        render_mode=None,
    )

    return env


def evaluate_agent(
    env: CrossChainEnv, agent, n_agents: int, n_episodes: int = 20, seed: int = 42
) -> Dict[str, List]:
    """
    Evaluate an agent over multiple episodes.

    Args:
        env: Environment
        agent: Agent to evaluate
        n_agents: Number of agents
        n_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "final_balances_eth": [],
        "final_balances_usdc": [],
        "gas_spent": [],
        "num_swaps": [],
        "num_bridges": [],
    }

    for episode in tqdm(range(n_episodes), desc="Evaluating"):
        obs, _ = env.reset(seed=seed + episode)

        if hasattr(agent, "reset_hidden_states"):
            agent.reset_hidden_states()

        episode_reward = 0.0
        done = False

        while not done:
            obs_dim = obs.shape[0] // n_agents
            agent_obs = [obs[i * obs_dim : (i + 1) * obs_dim] for i in range(n_agents)]

            if isinstance(agent, (QMIXAgent, MAPPOAgent)):
                actions = agent.select_actions(agent_obs, deterministic=True)
            else:
                actions = agent.select_actions(agent_obs)

            action = np.array([actions[0], 0, 100.0, 0])
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            done = terminated or truncated

        # Record metrics
        stats = env.get_stats()
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(stats["total_steps"])
        metrics["final_balances_eth"].append(stats["final_balances"].get("ETH", 0))
        metrics["final_balances_usdc"].append(stats["final_balances"].get("USDC", 0))
        metrics["gas_spent"].append(stats["total_gas_spent"])
        metrics["num_swaps"].append(stats["num_swaps"])
        metrics["num_bridges"].append(stats["num_bridges"])

    return metrics


def load_agent(
    checkpoint_path: str, agent_type: str, env: CrossChainEnv, n_agents: int = 2
):
    """Load a trained agent from checkpoint."""

    obs_dim = env.observation_space.shape[0] // n_agents
    state_dim = env.observation_space.shape[0]
    n_actions = 5

    if agent_type == "qmix":
        agent = QMIXAgent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            device="cpu",
        )
    elif agent_type == "mappo":
        agent = MAPPOAgent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            device="cpu",
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.load(checkpoint_path)
    return agent


def create_results_df(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a DataFrame from evaluation results."""

    rows = []

    for agent_name, metrics in results.items():
        for i in range(len(metrics["episode_rewards"])):
            rows.append(
                {
                    "agent": agent_name,
                    "episode": i,
                    "reward": metrics["episode_rewards"][i],
                    "length": metrics["episode_lengths"][i],
                    "final_eth": metrics["final_balances_eth"][i],
                    "final_usdc": metrics["final_balances_usdc"][i],
                    "gas_spent": metrics["gas_spent"][i],
                    "num_swaps": metrics["num_swaps"][i],
                    "num_bridges": metrics["num_bridges"][i],
                }
            )

    return pd.DataFrame(rows)


def plot_results(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization plots."""

    sns.set_style("whitegrid")

    # Plot 1: Episode Rewards
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Rewards over episodes
    ax = axes[0, 0]
    for agent in results_df["agent"].unique():
        agent_data = results_df[results_df["agent"] == agent]
        ax.plot(agent_data["episode"], agent_data["reward"], label=agent, alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reward distribution
    ax = axes[0, 1]
    results_df.boxplot(column="reward", by="agent", ax=ax)
    ax.set_title("Reward Distribution by Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Reward")

    # Gas spent
    ax = axes[1, 0]
    results_df.boxplot(column="gas_spent", by="agent", ax=ax)
    ax.set_title("Gas Spent by Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Gas Spent")

    # Number of swaps
    ax = axes[1, 1]
    results_df.boxplot(column="num_swaps", by="agent", ax=ax)
    ax.set_title("Number of Swaps by Agent")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Swaps")

    plt.tight_layout()
    plot_path = output_dir / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close()

    # Summary statistics plot
    fig, ax = plt.subplots(figsize=(10, 6))

    summary = (
        results_df.groupby("agent")
        .agg({"reward": ["mean", "std"], "gas_spent": "mean", "num_swaps": "mean"})
        .reset_index()
    )

    agents = summary["agent"].values
    means = summary[("reward", "mean")].values
    stds = summary[("reward", "std")].values

    x = np.arange(len(agents))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Episode Reward by Agent")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    summary_path = output_dir / "summary_plot.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"Summary plot saved: {summary_path}")
    plt.close()


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained MARL agents")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        required=True,
        choices=["qmix", "mappo"],
        help="Type of agent",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare against random baseline",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = create_test_env()

    print("=" * 60)
    print("Evaluation Setup")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Agent Type: {args.agent_type}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Output Directory: {output_dir}")

    # Load trained agent
    print("\nLoading trained agent...")
    try:
        trained_agent = load_agent(args.checkpoint, args.agent_type, env)
        print("Agent loaded successfully!")
    except Exception as e:
        print(f"Error loading agent: {e}")
        print("Using random agent for demonstration...")
        trained_agent = None

    # Evaluate agents
    results = {}
    n_agents = 2

    if trained_agent is not None:
        print(f"\nEvaluating {args.agent_type.upper()} agent...")
        metrics = evaluate_agent(env, trained_agent, n_agents, args.n_episodes)
        results[args.agent_type.upper()] = metrics

    if args.compare_baseline or trained_agent is None:
        print("\nEvaluating Random baseline...")
        random_agent = RandomAgent(n_agents=n_agents, n_actions=5)
        metrics = evaluate_agent(env, random_agent, n_agents, args.n_episodes)
        results["Random"] = metrics

    # Create results DataFrame
    print("\nCreating results DataFrame...")
    results_df = create_results_df(results)

    # Save results CSV
    csv_path = output_dir / "evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results CSV saved: {csv_path}")

    # Save summary statistics
    summary = results_df.groupby("agent").agg(
        {
            "reward": ["mean", "std", "min", "max"],
            "length": "mean",
            "gas_spent": "mean",
            "num_swaps": "mean",
            "num_bridges": "mean",
        }
    )

    summary_path = output_dir / "summary_statistics.json"
    summary_dict = summary.to_dict()

    # Convert to serializable format
    serializable_summary = {}
    for key, value in summary_dict.items():
        serializable_summary[str(key)] = {str(k): float(v) for k, v in value.items()}

    with open(summary_path, "w") as f:
        json.dump(serializable_summary, f, indent=2)
    print(f"Summary statistics saved: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(summary)

    # Create plots
    print("\nCreating visualization plots...")
    plot_results(results_df, output_dir)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
