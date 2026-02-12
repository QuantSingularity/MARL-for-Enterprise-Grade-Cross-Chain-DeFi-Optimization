"""
Evaluation Metrics for MARL Agents

Comprehensive metrics for evaluating DeFi optimization performance.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    """Container for episode-level metrics."""

    episode_reward: float
    episode_length: int
    total_gas_spent: float
    num_swaps: int
    num_bridges: int
    avg_price_impact: float
    total_slippage: float
    capital_efficiency: float
    sharpe_ratio: float
    final_portfolio_value: float


class MetricsCalculator:
    """Calculate comprehensive performance metrics."""

    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float], risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sharpe ratio of returns."""
        if len(returns) < 2:
            return 0.0
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns)

    @staticmethod
    def calculate_max_drawdown(cumulative_rewards: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_rewards) < 2:
            return 0.0
        cumulative = np.array(cumulative_rewards)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        return float(np.min(drawdown))

    @staticmethod
    def calculate_capital_efficiency(
        initial_capital: float, final_capital: float, locked_capital: float
    ) -> float:
        """Calculate capital efficiency ratio."""
        if locked_capital == 0:
            return 0.0
        profit = final_capital - initial_capital
        return profit / locked_capital

    @staticmethod
    def calculate_win_rate(rewards: List[float]) -> float:
        """Calculate percentage of positive reward episodes."""
        if len(rewards) == 0:
            return 0.0
        positive = sum(1 for r in rewards if r > 0)
        return positive / len(rewards)

    @staticmethod
    def calculate_volatility(returns: List[float]) -> float:
        """Calculate volatility (std dev) of returns."""
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns))

    @staticmethod
    def aggregate_metrics(episode_data_list: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across multiple episodes."""
        rewards = [ep["episode_reward"] for ep in episode_data_list]
        lengths = [ep["episode_length"] for ep in episode_data_list]

        cumulative_rewards = np.cumsum(rewards)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "median_reward": float(np.median(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "mean_length": float(np.mean(lengths)),
            "sharpe_ratio": MetricsCalculator.calculate_sharpe_ratio(rewards),
            "max_drawdown": MetricsCalculator.calculate_max_drawdown(
                cumulative_rewards.tolist()
            ),
            "win_rate": MetricsCalculator.calculate_win_rate(rewards),
            "volatility": MetricsCalculator.calculate_volatility(rewards),
            "total_episodes": len(episode_data_list),
        }
