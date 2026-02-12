# MARL for Cross-Chain DeFi Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Production-grade implementation of **Multi-Agent Reinforcement Learning (MARL)** for optimizing cross-chain decentralized finance (DeFi) operations. This system addresses liquidity fragmentation, suboptimal routing, and bridge risks across multiple blockchain networks using state-of-the-art MARL algorithms.

## Quick Start

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/quantsingularity/marl-crosschain-defi.git
cd marl-crosschain-defi/code

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (~10 minutes on CPU)
./cli.sh all
```

### Option 2: Docker

```bash
# Build Docker image
docker build -t marl-defi .

# Run training
docker-compose up marl-training

# Run evaluation
docker-compose up marl-eval
```

### Option 3: Quick Demo

```bash
cd code

# Test environment
python src/envs/demo_env.py --steps 20

# Train QMIX
python src/train/train_synthetic.py --agent qmix --config configs/demo.yaml

# Evaluate
python src/eval/evaluate_demo.py --checkpoint checkpoints/qmix_model.pth --agent-type qmix
```

## Architecture

```
Cross-Chain DeFi Environment
├── Multiple Blockchains (Ethereum, Arbitrum, ...)
├── Cross-Chain Bridges (with latency & risk)
├── DEX Liquidity Pools (AMM mechanics)
└── Gas Costs & Transaction Execution

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

## Repository Structure

```
marl-crosschain-defi/
├── code/
│   ├── src/
│   │   ├── agents/          # MARL algorithms (QMIX, MAPPO, baselines)
│   │   │   ├── qmix.py
│   │   │   ├── mappo.py
│   │   │   ├── baselines.py
│   │   │   ├── communication.py  # Agent communication modules
│   │   │   └── gnn_encoder.py    # Graph neural network encoders
│   │   ├── envs/            # Cross-chain environment simulator
│   │   │   ├── cross_chain_env.py
│   │   │   └── demo_env.py
│   │   ├── train/           # Training scripts
│   │   │   └── train_synthetic.py
│   │   ├── eval/            # Evaluation scripts
│   │   │   ├── evaluate_demo.py
│   │   │   └── metrics.py
│   │   ├── data/            # Data generation
│   │   │   └── generate_synthetic.py
│   │   └── utils/           # Utilities (logging, checkpointing)
│   │       ├── logger.py
│   │       ├── checkpointing.py
│   │       └── config.py
│   ├── tests/               # Comprehensive test suite
│   │   ├── test_env.py
│   │   ├── test_agents.py
│   │   ├── test_utils.py
│   │   └── test_integration.py
│   ├── configs/             # Configuration files
│   │   └── demo.yaml
│   ├── notebooks/           # Jupyter notebooks
│   │   ├── 01_environment_demo.ipynb
│   │   └── 02_training_demo.ipynb
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── cli.sh               # Command-line interface
├── docs/                    # Documentation
│   ├── required_resources.md
│   └── DATA_ACQUISITION_PLAN.md
├── .github/
│   └── workflows/           # CI/CD pipelines
│       ├── cicd.yml         # Formatting checks
│       └── test.yml         # Automated testing
├── Dockerfile
├── docker-compose.yml
├── README.md
└── LICENSE
```

## Algorithms Implemented

### QMIX (Monotonic Value Function Factorization)

- **Paper**: [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- **Key Features**:
  - Decentralized execution with centralized training (CTDE)
  - Monotonic mixing network for credit assignment
  - DRQN-based agent networks with GRU cells
  - Experience replay buffer
  - Target network soft updates

### MAPPO (Multi-Agent Proximal Policy Optimization)

- **Paper**: [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- **Key Features**:
  - Shared parameter actor network
  - Centralized value function
  - PPO clipping for stable updates
  - Generalized Advantage Estimation (GAE)
  - Multiple update epochs per rollout

### Communication Modules

- **CommNet**: Iterative message passing between agents
- **Attention-based Communication**: Multi-head attention for selective messaging
- **GNN Encoders**: Graph neural networks for agent state representation

## Configuration

Edit `code/configs/demo.yaml` to customize:

```yaml
environment:
  chains: # Blockchain networks
    - name: Ethereum
      chain_id: 1
      block_time: 12.0
      base_gas_price: 20.0
  bridges: # Cross-chain bridges
    - name: Arbitrum_Bridge
      latency_mean: 600.0
      failure_rate: 0.01
  pools: # DEX liquidity pools
    Ethereum:
      - token_a: ETH
        token_b: USDC
        reserve_a: 1000.0

training:
  n_agents: 2
  hidden_dim: 64
  gamma: 0.99
  n_episodes: 50
  device: cpu # Use 'cuda' for GPU
```

## Testing

```bash
# Run all tests
pytest code/tests/ -v

# Run with coverage
pytest code/tests/ --cov=code/src --cov-report=html

# Run specific test file
pytest code/tests/test_agents.py -v

# Run integration tests
pytest code/tests/test_integration.py -v
```

## Performance

### Benchmark Results (50 episodes, demo config)

| Agent                  | Mean Reward | Std Dev | Episode Length | Training Time |
| ---------------------- | ----------- | ------- | -------------- | ------------- |
| Random                 | -15.2       | 5.3     | 50.0           | -             |
| Independent Q-Learning | -8.4        | 4.1     | 48.2           | ~5 min        |
| **QMIX**               | **-2.1**    | **3.2** | **45.7**       | **~8 min**    |
| **MAPPO**              | **-1.8**    | **2.9** | **44.3**       | **~10 min**   |

_Results on Intel i7 CPU with demo configuration_

## CLI Commands

The `cli.sh` script provides convenient commands:

```bash
./cli.sh all       # Run complete pipeline
./cli.sh data      # Generate synthetic data
./cli.sh train     # Train agents
./cli.sh eval      # Evaluate agents
./cli.sh test      # Run tests
./cli.sh demo      # Run environment demo
./cli.sh clean     # Clean generated files
```

## Advanced Usage

### Custom Training

```python
from pathlib import Path
from src.envs.cross_chain_env import CrossChainEnv, Chain, Bridge, Pool
from src.agents.qmix import QMIXAgent
from src.utils.logger import MetricsLogger
from src.utils.checkpointing import CheckpointManager

# Create environment
env = CrossChainEnv(chains=[...], bridges=[...], pools={...})

# Create agent
agent = QMIXAgent(
    n_agents=2,
    obs_dim=env.observation_space.shape[0] // 2,
    state_dim=env.observation_space.shape[0],
    n_actions=5,
    hidden_dim=128,
    device="cuda"
)

# Setup logging
logger = MetricsLogger(Path("logs/custom_run"))
checkpoint_mgr = CheckpointManager(Path("checkpoints/custom"))

# Training loop
for episode in range(num_episodes):
    # ... your training code ...
    logger.log_dict(metrics, episode)
    checkpoint_mgr.save_checkpoint(agent, episode, metrics)
```

### Adding New Agents

1. Create agent file in `code/src/agents/`
2. Implement required methods: `select_actions()`, `update()`, `save()`, `load()`
3. Add agent to training script
4. Write tests in `code/tests/test_agents.py`

## Docker Usage

### Build and Run

```bash
# Build image
docker build -t marl-defi:latest .

# Run training
docker run -v $(pwd)/checkpoints:/app/checkpoints marl-defi \
  python code/src/train/train_synthetic.py --agent qmix

# Run with custom config
docker run -v $(pwd)/checkpoints:/app/checkpoints \
           -v $(pwd)/configs:/app/configs \
           marl-defi \
  python code/src/train/train_synthetic.py --config configs/custom.yaml

# Run tests
docker run marl-defi pytest code/tests/ -v
```

### Docker Compose

```bash
# Train all agents
docker-compose up marl-training

# Run evaluation
docker-compose up marl-eval

# View logs
docker-compose logs -f marl-training
```

## Troubleshooting

### Common Issues

**Import Errors**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/code/src"
```

**Out of Memory**

- Reduce `batch_size` in config
- Reduce `hidden_dim`
- Use CPU instead of GPU for demo

**Slow Training**

- Reduce `n_episodes`
- Use GPU: `device: cuda` in config
- Reduce environment complexity

**Tests Failing**

```bash
cd code
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

### Development

```bash
# Install dev dependencies
pip install -r code/requirements.txt
pip install black isort flake8 pytest-cov

# Format code
black code/
isort code/

# Run linters
flake8 code/src/

# Run tests with coverage
pytest code/tests/ --cov=code/src --cov-report=html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
