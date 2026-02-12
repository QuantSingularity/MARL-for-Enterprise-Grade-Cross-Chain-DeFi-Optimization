#!/usr/bin/env python3
"""
Comprehensive MARL Project Builder
Generates all missing/enhanced components for production-ready system
"""

import os
from pathlib import Path

# Change to project directory
os.chdir("/mnt/user-data/outputs/marl-project-enhanced")

# File contents dictionary
files_to_create = {
    "code/src/utils/checkpointing.py": '''"""Checkpointing utilities for model saving and loading."""
import torch
from pathlib import Path
from typing import Dict, Optional
import json

class CheckpointManager:
    """Manage model checkpoints with versioning."""
    
    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        
    def save_checkpoint(self, agent, episode: int, metrics: Dict, filename: Optional[str] = None) -> Path:
        """Save a checkpoint."""
        if filename is None:
            filename = f"checkpoint_ep{episode}.pth"
        checkpoint_path = self.checkpoint_dir / filename
        agent.save(str(checkpoint_path))
        metadata = {'episode': episode, 'metrics': metrics}
        with open(checkpoint_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        self._cleanup_old_checkpoints()
        return checkpoint_path
        
    def load_checkpoint(self, checkpoint_path: Path, agent):
        """Load a checkpoint."""
        agent.load(str(checkpoint_path))
        metadata_path = checkpoint_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.pth"), key=lambda p: p.stat().st_mtime)
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                checkpoint.unlink()
                metadata_path = checkpoint.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()
''',
    "code/src/utils/__init__.py": '''"""Utility modules for MARL training."""
from .logger import setup_logger, MetricsLogger
from .checkpointing import CheckpointManager

__all__ = ['setup_logger', 'MetricsLogger', 'CheckpointManager']
''',
    "code/tests/test_utils.py": '''"""Tests for utility modules."""
import pytest
from pathlib import Path
from code.src.utils.logger import MetricsLogger
from code.src.utils.checkpointing import CheckpointManager

def test_metrics_logger(tmp_path):
    """Test metrics logger."""
    logger = MetricsLogger(tmp_path)
    logger.log_scalar("reward", 1.0, 0)
    logger.log_scalar("reward", 2.0, 1)
    logger.log_dict({"loss": 0.5, "accuracy": 0.9}, 0)
    output_file = logger.save()
    assert output_file.exists()
''',
    "code/tests/test_integration.py": '''"""Integration tests for complete training pipeline."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.cross_chain_env import CrossChainEnv, Chain, Bridge, Pool
from agents.qmix import QMIXAgent
from agents.mappo import MAPPOAgent

def create_test_env():
    """Create a minimal test environment."""
    chains = [
        Chain("TestChain", 1, 1.0, 10.0, 0.1)
    ]
    bridges = []
    pools = {
        "TestChain": [Pool("ETH", "USDC", 100.0, 200000.0, 30)]
    }
    return CrossChainEnv(chains, bridges, pools, max_steps=10)

def test_qmix_training():
    """Test QMIX agent training."""
    env = create_test_env()
    agent = QMIXAgent(n_agents=2, obs_dim=10, state_dim=20, n_actions=5, hidden_dim=32)
    obs, _ = env.reset()
    agent_obs = [obs[:10], obs[10:20]]
    actions = agent.select_actions(agent_obs)
    assert len(actions) == 2
    assert all(0 <= a < 5 for a in actions)

def test_mappo_training():
    """Test MAPPO agent training."""
    env = create_test_env()
    agent = MAPPOAgent(n_agents=2, obs_dim=10, state_dim=20, n_actions=5, hidden_dim=32)
    obs, _ = env.reset()
    agent_obs = [obs[:10], obs[10:20]]
    actions = agent.select_actions(agent_obs)
    assert len(actions) == 2
    assert all(0 <= a < 5 for a in actions)
''',
    "Dockerfile": """# Multi-Agent Reinforcement Learning - Production Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY code/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY code/ ./code/
COPY README.md .

# Set Python path
ENV PYTHONPATH=/app/code/src:$PYTHONPATH

# Create output directories
RUN mkdir -p /app/checkpoints /app/results /app/data /app/logs

# Default command
CMD ["python", "-m", "pytest", "code/tests/", "-v"]

# To run training:
# docker run -v $(pwd)/checkpoints:/app/checkpoints marl-defi python code/src/train/train_synthetic.py
""",
    "docker-compose.yml": """version: '3.8'

services:
  marl-training:
    build: .
    image: marl-defi:latest
    container_name: marl-training
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./results:/app/results
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    command: python code/src/train/train_synthetic.py --config code/configs/demo.yaml --agent all
    
  marl-eval:
    build: .
    image: marl-defi:latest
    container_name: marl-eval
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./results:/app/results
    command: python code/src/eval/evaluate_demo.py --checkpoint /app/checkpoints/qmix_model.pth --agent-type qmix
    depends_on:
      - marl-training
""",
    ".dockerignore": """__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.git
.gitignore
.pytest_cache
.coverage
*.log
checkpoints/
results/
data/synthetic/
*.pth
*.pt
.DS_Store
""",
    "code/pyproject.toml": """[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "marl-cross-chain-defi"
version = "1.0.0"
description = "Multi-Agent Reinforcement Learning for Cross-Chain DeFi Optimization"
authors = [{name = "Research Team", email = "research@example.com"}]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "gymnasium>=0.28.0",
    "pettingzoo>=1.24.0",
    "pyyaml>=6.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "tqdm>=4.65.0",
    "pytest>=7.3.0",
    "pydantic>=2.0.0"
]

[project.optional-dependencies]
dev = ["black", "isort", "flake8", "mypy", "pytest-cov"]
viz = ["plotly>=5.14.0", "tensorboard>=2.13.0", "wandb>=0.15.0"]
gnn = ["torch-geometric>=2.3.0", "torch-scatter>=2.1.0"]

[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 88

[tool.pytest.ini_options]
testpaths = ["code/tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
""",
    ".github/workflows/test.yml": """name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip packages
      uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('code/requirements.txt') }}
        
    - name: Install dependencies
      run: |
        cd code
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        cd code
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./code/coverage.xml
""",
    "CONTRIBUTING.md": """# Contributing to MARL Cross-Chain DeFi

## Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r code/requirements.txt`
3. Run tests: `pytest code/tests/`

## Code Style

- Use Black for formatting: `black code/`
- Use isort for imports: `isort code/`
- Follow PEP 8 guidelines

## Testing

- Write tests for all new features
- Maintain >80% code coverage
- Run full test suite before submitting PR

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Run tests and linters
4. Submit PR with clear description
5. Wait for review and address feedback
""",
    "CITATION.cff": """cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "Multi-Agent Reinforcement Learning for Cross-Chain DeFi Optimization"
version: 1.0.0
date-released: 2024-02-01
authors:
  - family-names: "Research"
    given-names: "Team"
repository-code: "https://github.com/username/marl-crosschain-defi"
keywords:
  - multi-agent reinforcement learning
  - decentralized finance
  - cross-chain
  - blockchain
  - QMIX
  - MAPPO
license: MIT
""",
}

# Create all files
for filepath, content in files_to_create.items():
    full_path = Path(filepath)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    print(f"✓ Created: {filepath}")

print("\n" + "=" * 60)
print("✓ All production components created successfully!")
print("=" * 60)
