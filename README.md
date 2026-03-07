# MARL for Cross-Chain DeFi Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-grade **Multi-Agent Reinforcement Learning (MARL)** for optimizing cross-chain DeFi operations. Addresses liquidity fragmentation, suboptimal routing, and bridge risks across multiple blockchain networks using QMIX and MAPPO with centralized training and decentralized execution.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Algorithms](#algorithms)
- [Configuration](#configuration)
- [Results](#results)
- [CLI Reference](#cli-reference)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

| Component           | Description                                                                              |
| :------------------ | :--------------------------------------------------------------------------------------- |
| **MARL Algorithms** | QMIX (value decomposition) and MAPPO (policy optimization) with CTDE paradigm            |
| **Environment**     | Cross-chain simulator with AMM mechanics, bridge latency, gas costs, and liquidity pools |
| **Communication**   | CommNet, attention-based messaging, and GNN encoders for agent state representation      |
| **Baselines**       | Random policy and Independent Q-Learning for benchmarking                                |
| **Infrastructure**  | Docker, CI/CD, MLflow-compatible logging, and checkpoint management                      |

---

## Architecture

```
Cross-Chain DeFi Environment
├── Multiple Blockchains (Ethereum, Arbitrum, ...)
├── Cross-Chain Bridges (latency and risk modeling)
├── DEX Liquidity Pools (AMM mechanics)
└── Gas Costs and Transaction Execution

MARL Agents
├── QMIX (Value Decomposition)
│   ├── Agent Networks (DRQN)
│   ├── Mixing Network
│   └── Centralized Training, Decentralized Execution
├── MAPPO (Policy Optimization)
│   ├── Actor Networks (shared parameters)
│   ├── Centralized Critic
│   └── PPO Updates with GAE
└── Baselines (Random, Independent Q-Learning)
```

---

## Quick Start

### Local

```bash
git clone https://github.com/quantsingularity/MARL-for-Enterprise-Grade-Cross-Chain-DeFi-Optimization
cd MARL-for-Enterprise-Grade-Cross-Chain-DeFi-Optimization/code

pip install -r requirements.txt
./cli.sh all
```

### Docker

```bash
docker build -t marl-defi .
docker-compose up marl-training
docker-compose up marl-eval
```

### Quick Demo

```bash
cd code
python src/envs/demo_env.py --steps 20
python src/train/train_synthetic.py --agent qmix --config configs/demo.yaml
python src/eval/evaluate_demo.py --checkpoint checkpoints/qmix_model.pth --agent-type qmix
```

---

## Repository Structure

```
code/
├── src/
│   ├── agents/       # QMIX, MAPPO, baselines, communication, GNN encoders
│   ├── envs/         # Cross-chain environment simulator
│   ├── train/        # Training scripts
│   ├── eval/         # Evaluation scripts and metrics
│   ├── data/         # Synthetic data generation
│   └── utils/        # Logging, checkpointing, config
├── tests/            # Unit and integration tests
├── configs/          # YAML configuration files
├── notebooks/        # Environment and training demos
├── requirements.txt
└── cli.sh
```

---

## Algorithms

### QMIX

Monotonic value function factorization for cooperative MARL. [Paper](https://arxiv.org/abs/1803.11485)

- Centralized training with decentralized execution (CTDE)
- Monotonic mixing network for credit assignment
- DRQN-based agent networks with GRU cells
- Experience replay and target network soft updates

### MAPPO

Multi-Agent Proximal Policy Optimization for cooperative tasks. [Paper](https://arxiv.org/abs/2103.01955)

- Shared parameter actor network
- Centralized value function
- PPO clipping for stable updates
- Generalized Advantage Estimation (GAE)

### Communication Modules

- **CommNet:** Iterative message passing between agents
- **Attention-based:** Multi-head attention for selective messaging
- **GNN Encoders:** Graph neural network state representation

---

## Configuration

Edit `code/configs/demo.yaml` to customize the environment and training.

```yaml
environment:
  chains:
    - name: Ethereum
      chain_id: 1
      block_time: 12.0
      base_gas_price: 20.0
  bridges:
    - name: Arbitrum_Bridge
      latency_mean: 600.0
      failure_rate: 0.01
  pools:
    Ethereum:
      - token_a: ETH
        token_b: USDC
        reserve_a: 1000.0

training:
  n_agents: 2
  hidden_dim: 64
  gamma: 0.99
  n_episodes: 50
  device: cpu
```

---

## Results

Benchmark results over 50 episodes on Intel i7 CPU with demo configuration.

| Agent                  | Mean Reward | Std Dev | Episode Length | Training Time |
| :--------------------- | :---------- | :------ | :------------- | :------------ |
| Random                 | -15.2       | 5.3     | 50.0           | -             |
| Independent Q-Learning | -8.4        | 4.1     | 48.2           | ~5 min        |
| **QMIX**               | **-2.1**    | **3.2** | **45.7**       | ~8 min        |
| **MAPPO**              | **-1.8**    | **2.9** | **44.3**       | ~10 min       |

---

## CLI Reference

```bash
./cli.sh all       # Complete pipeline
./cli.sh data      # Generate synthetic data
./cli.sh train     # Train agents
./cli.sh eval      # Evaluate agents
./cli.sh test      # Run tests
./cli.sh demo      # Environment demo
./cli.sh clean     # Clean generated files
```

---

## Advanced Usage

### Custom Training

```python
from src.envs.cross_chain_env import CrossChainEnv
from src.agents.qmix import QMIXAgent
from src.utils.logger import MetricsLogger
from src.utils.checkpointing import CheckpointManager
from pathlib import Path

env = CrossChainEnv(chains=[...], bridges=[...], pools={...})

agent = QMIXAgent(
    n_agents=2,
    obs_dim=env.observation_space.shape[0] // 2,
    state_dim=env.observation_space.shape[0],
    n_actions=5,
    hidden_dim=128,
    device="cuda"
)

logger = MetricsLogger(Path("logs/custom_run"))
checkpoint_mgr = CheckpointManager(Path("checkpoints/custom"))

for episode in range(num_episodes):
    logger.log_dict(metrics, episode)
    checkpoint_mgr.save_checkpoint(agent, episode, metrics)
```

### Adding New Agents

1. Create agent file in `code/src/agents/`
2. Implement `select_actions()`, `update()`, `save()`, `load()`
3. Register agent in the training script
4. Write tests in `code/tests/test_agents.py`

### Testing

```bash
pytest code/tests/ -v
pytest code/tests/ --cov=code/src --cov-report=html
```

---

## Troubleshooting

| Issue         | Fix                                                                         |
| :------------ | :-------------------------------------------------------------------------- |
| Import errors | `export PYTHONPATH="${PYTHONPATH}:$(pwd)/code/src"`                         |
| Out of memory | Reduce `batch_size` or `hidden_dim` in config                               |
| Slow training | Set `device: cuda` in config or reduce `n_episodes`                         |
| Tests failing | `cd code && pip install -r requirements.txt && pytest tests/ -v --tb=short` |

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
