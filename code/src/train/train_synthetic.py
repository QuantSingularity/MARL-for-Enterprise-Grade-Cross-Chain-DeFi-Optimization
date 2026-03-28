"""
Training Script for MARL Agents on Synthetic Cross-Chain Environment

Demonstrates end-to-end training pipeline with QMIX and MAPPO agents.
Designed to run quickly on CPU for demonstration purposes.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from envs.cross_chain_env import CrossChainEnv, Chain, Bridge, Pool
from agents.qmix import QMIXAgent
from agents.mappo import MAPPOAgent
from agents.baselines import IndependentQLearning


class ReplayBuffer:
    """Simple replay buffer for MARL."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition: Dict):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict:
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        # Stack tensors
        return {
            key: torch.stack([torch.FloatTensor(b[key]) for b in batch])
            for key in batch[0].keys()
        }

    def __len__(self):
        return len(self.buffer)


def create_synthetic_env(config: Dict) -> CrossChainEnv:
    """Create environment from config."""

    # Create chains
    chains = []
    for chain_config in config["chains"]:
        chains.append(
            Chain(
                name=chain_config["name"],
                chain_id=chain_config["chain_id"],
                block_time=chain_config["block_time"],
                base_gas_price=chain_config["base_gas_price"],
                gas_volatility=chain_config["gas_volatility"],
            )
        )

    # Create bridges
    bridges = []
    for bridge_config in config["bridges"]:
        bridges.append(
            Bridge(
                name=bridge_config["name"],
                source_chain=bridge_config["source_chain"],
                target_chain=bridge_config["target_chain"],
                capacity=bridge_config["capacity"],
                latency_mean=bridge_config["latency_mean"],
                latency_std=bridge_config["latency_std"],
                failure_rate=bridge_config["failure_rate"],
                fee_basis_points=bridge_config["fee_basis_points"],
            )
        )

    # Create pools
    pools = {}
    for chain_name, pool_configs in config["pools"].items():
        pools[chain_name] = []
        for pool_config in pool_configs:
            pools[chain_name].append(
                Pool(
                    token_a=pool_config["token_a"],
                    token_b=pool_config["token_b"],
                    reserve_a=pool_config["reserve_a"],
                    reserve_b=pool_config["reserve_b"],
                    fee_tier=pool_config["fee_tier"],
                )
            )

    return CrossChainEnv(
        chains=chains,
        bridges=bridges,
        pools=pools,
        initial_balances=config.get("initial_balances", {"ETH": 10.0, "USDC": 20000.0}),
        max_steps=config.get("max_steps", 100),
    )


def rollout_episode(
    env: CrossChainEnv, agent, n_agents: int, render: bool = False
) -> Dict[str, float]:
    """
    Rollout a single episode.

    Args:
        env: Environment
        agent: Agent to evaluate
        n_agents: Number of agents
        render: Whether to render

    Returns:
        Episode statistics
    """
    obs, _ = env.reset()

    # Reset agent hidden states if applicable
    if hasattr(agent, "reset_hidden_states"):
        agent.reset_hidden_states()

    episode_reward = 0.0
    episode_length = 0

    # Storage for training
    observations = []
    actions = []
    rewards = []
    states = []

    done = False

    while not done:
        # Split observation for each agent (simplified)
        obs_dim = obs.shape[0] // n_agents
        agent_obs = [obs[i * obs_dim : (i + 1) * obs_dim] for i in range(n_agents)]

        # Select actions
        if isinstance(agent, (QMIXAgent, IndependentQLearning)):
            actions_list = agent.select_actions(agent_obs)
        elif isinstance(agent, MAPPOAgent):
            actions_list = agent.select_actions(agent_obs)
        else:  # RandomAgent
            actions_list = agent.select_actions(agent_obs)

        # Execute action (use first agent's action for single-agent env)
        action = np.array([actions_list[0], 0, 100.0, 0])
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store transition
        observations.append(obs)
        actions.append(actions_list)
        rewards.append(reward)
        states.append(obs)  # Using obs as state for simplicity

        episode_reward += reward
        episode_length += 1

        obs = next_obs
        done = terminated or truncated

        if render:
            env.render()

    return {
        "episode_reward": episode_reward,
        "episode_length": episode_length,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "states": states,
    }


def train_qmix(
    env: CrossChainEnv, config: Dict, save_dir: Path
) -> Dict[str, List[float]]:
    """Train QMIX agent."""

    print("\n" + "=" * 60)
    print("Training QMIX Agent")
    print("=" * 60)

    n_agents = config["n_agents"]
    obs_dim = env.observation_space.shape[0] // n_agents
    state_dim = env.observation_space.shape[0]
    n_actions = 5  # From action space

    agent = QMIXAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=config.get("hidden_dim", 64),
        gamma=config.get("gamma", 0.99),
        lr=config.get("lr", 5e-4),
        device=config.get("device", "cpu"),
    )

    buffer = ReplayBuffer(capacity=config.get("buffer_size", 5000))

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "losses": [],
        "epsilons": [],
    }

    n_episodes = config.get("n_episodes", 100)
    batch_size = config.get("batch_size", 32)
    target_update_freq = config.get("target_update_freq", 10)

    for episode in tqdm(range(n_episodes), desc="Training QMIX"):
        # Rollout episode
        episode_data = rollout_episode(env, agent, n_agents)

        metrics["episode_rewards"].append(episode_data["episode_reward"])
        metrics["episode_lengths"].append(episode_data["episode_length"])

        # Add to buffer
        for t in range(len(episode_data["observations"]) - 1):
            transition = {
                "observations": np.array(episode_data["observations"][t]),
                "actions": np.array(episode_data["actions"][t]),
                "rewards": np.array([episode_data["rewards"][t]]),
                "next_observations": np.array(episode_data["observations"][t + 1]),
                "states": np.array(episode_data["states"][t]),
                "next_states": np.array(episode_data["states"][t + 1]),
                "dones": np.array(
                    [1.0 if t == len(episode_data["observations"]) - 2 else 0.0]
                ),
            }
            buffer.push(transition)

        # Train if enough data
        if len(buffer) >= batch_size:
            for _ in range(config.get("updates_per_episode", 1)):
                batch = buffer.sample(batch_size)

                # Reshape for QMIX
                batch["observations"] = batch["observations"].view(
                    batch_size, n_agents, -1
                )
                batch["actions"] = batch["actions"].long()
                batch["next_observations"] = batch["next_observations"].view(
                    batch_size, n_agents, -1
                )

                train_metrics = agent.update(batch)
                metrics["losses"].append(train_metrics["loss"])
                metrics["epsilons"].append(train_metrics["epsilon"])

        # Update target networks
        if episode % target_update_freq == 0:
            agent.update_target_networks(tau=0.01)

        # Log progress
        if episode % 10 == 0:
            avg_reward = np.mean(metrics["episode_rewards"][-10:])
            print(
                f"Episode {episode}: Avg Reward = {avg_reward:.4f}, Epsilon = {agent.epsilon:.4f}"
            )

    # Save model
    model_path = save_dir / "qmix_model.pth"
    agent.save(str(model_path))
    print(f"Model saved: {model_path}")

    return metrics


def train_mappo(
    env: CrossChainEnv, config: Dict, save_dir: Path
) -> Dict[str, List[float]]:
    """Train MAPPO agent."""

    print("\n" + "=" * 60)
    print("Training MAPPO Agent")
    print("=" * 60)

    n_agents = config["n_agents"]
    obs_dim = env.observation_space.shape[0] // n_agents
    state_dim = env.observation_space.shape[0]
    n_actions = 5

    agent = MAPPOAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=config.get("hidden_dim", 64),
        lr_actor=config.get("lr_actor", 3e-4),
        lr_critic=config.get("lr_critic", 1e-3),
        gamma=config.get("gamma", 0.99),
        device=config.get("device", "cpu"),
    )

    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "actor_losses": [],
        "critic_losses": [],
    }

    n_episodes = config.get("n_episodes", 100)

    for episode in tqdm(range(n_episodes), desc="Training MAPPO"):
        # Collect trajectory
        obs, _ = env.reset()

        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "states": [],
            "dones": [],
            "old_log_probs": [],
        }

        done = False
        episode_reward = 0.0

        while not done:
            obs_dim_split = obs.shape[0] // n_agents
            agent_obs = [
                obs[i * obs_dim_split : (i + 1) * obs_dim_split]
                for i in range(n_agents)
            ]

            actions = agent.select_actions(agent_obs)

            action = np.array([actions[0], 0, 100.0, 0])
            next_obs, reward, terminated, truncated, info = env.step(action)

            trajectory["observations"].append(obs)
            trajectory["actions"].append(actions)
            trajectory["rewards"].append(reward)
            trajectory["states"].append(obs)
            trajectory["dones"].append(float(terminated or truncated))
            trajectory["old_log_probs"].append(np.zeros(n_agents))  # Placeholder

            episode_reward += reward
            obs = next_obs
            done = terminated or truncated

        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(len(trajectory["rewards"]))

        # Prepare batch for update
        batch = {
            "observations": torch.FloatTensor(
                np.array(trajectory["observations"])
            ).view(-1, n_agents, obs_dim_split),
            "actions": torch.LongTensor(np.array(trajectory["actions"])),
            "rewards": torch.FloatTensor(trajectory["rewards"]),
            "states": torch.FloatTensor(np.array(trajectory["states"])),
            "dones": torch.FloatTensor(trajectory["dones"]),
            "old_log_probs": torch.FloatTensor(np.array(trajectory["old_log_probs"])),
        }

        # Update
        if len(trajectory["rewards"]) > 0:
            train_metrics = agent.update(batch, n_epochs=config.get("ppo_epochs", 5))
            metrics["actor_losses"].append(train_metrics["actor_loss"])
            metrics["critic_losses"].append(train_metrics["critic_loss"])

        if episode % 10 == 0:
            avg_reward = np.mean(metrics["episode_rewards"][-10:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.4f}")

    # Save model
    model_path = save_dir / "mappo_model.pth"
    agent.save(str(model_path))
    print(f"Model saved: {model_path}")

    return metrics


def _to_serializable(vals):
    """
    Convert a list of metric values to a JSON-serialisable list of Python floats.

    """
    if not vals:
        return vals
    if isinstance(vals[0], (float, int)):
        return vals
    # Handle numpy scalar types
    return [float(v) for v in vals]


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train MARL agents on synthetic environment"
    )
    parser.add_argument(
        "--config", type=str, default="configs/demo.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="qmix",
        choices=["qmix", "mappo", "all"],
        help="Agent type to train",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for models",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found: {config_path}, using default config")
        config = get_default_config()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = create_synthetic_env(config["environment"])

    print("Environment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Train agents
    all_metrics = {}

    if args.agent in ["qmix", "all"]:
        metrics = train_qmix(env, config["training"], output_dir)
        all_metrics["qmix"] = metrics

    if args.agent in ["mappo", "all"]:
        metrics = train_mappo(env, config["training"], output_dir)
        all_metrics["mappo"] = metrics

    # Save metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:

        serializable_metrics = {
            agent_name: {k: _to_serializable(vals) for k, vals in m.items()}
            for agent_name, m in all_metrics.items()
        }
        json.dump(serializable_metrics, f, indent=2)

    print(f"\nMetrics saved: {metrics_path}")
    print("Training complete!")


def get_default_config() -> Dict:
    """Get default configuration."""
    return {
        "environment": {
            "chains": [
                {
                    "name": "Ethereum",
                    "chain_id": 1,
                    "block_time": 12.0,
                    "base_gas_price": 20.0,
                    "gas_volatility": 0.1,
                },
                {
                    "name": "Arbitrum",
                    "chain_id": 42161,
                    "block_time": 0.25,
                    "base_gas_price": 0.1,
                    "gas_volatility": 0.05,
                },
            ],
            "bridges": [
                {
                    "name": "Arbitrum_Bridge",
                    "source_chain": "Ethereum",
                    "target_chain": "Arbitrum",
                    "capacity": 10000.0,
                    "latency_mean": 600.0,
                    "latency_std": 120.0,
                    "failure_rate": 0.01,
                    "fee_basis_points": 10,
                }
            ],
            "pools": {
                "Ethereum": [
                    {
                        "token_a": "ETH",
                        "token_b": "USDC",
                        "reserve_a": 1000.0,
                        "reserve_b": 2000000.0,
                        "fee_tier": 30,
                    }
                ],
                "Arbitrum": [
                    {
                        "token_a": "ETH",
                        "token_b": "USDC",
                        "reserve_a": 500.0,
                        "reserve_b": 1000000.0,
                        "fee_tier": 30,
                    }
                ],
            },
            "initial_balances": {"ETH": 10.0, "USDC": 20000.0},
            "max_steps": 50,
        },
        "training": {
            "n_agents": 2,
            "hidden_dim": 64,
            "gamma": 0.99,
            "lr": 5e-4,
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "n_episodes": 50,
            "batch_size": 16,
            "buffer_size": 2000,
            "target_update_freq": 5,
            "updates_per_episode": 1,
            "ppo_epochs": 5,
            "device": "cpu",
        },
    }


if __name__ == "__main__":
    main()
