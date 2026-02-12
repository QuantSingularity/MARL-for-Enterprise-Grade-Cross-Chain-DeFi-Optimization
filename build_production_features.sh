#!/bin/bash
# Build Production Features Script
set -e

cd /mnt/user-data/outputs/marl-project-enhanced

echo "Building production-grade features..."

# Create logger module
cat > code/src/utils/logger.py << 'EOF'
"""
Logging utilities for MARL training and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """Setup logger with file and console handlers."""
    
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    formatter = logging.Formatter(format_str)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for training metrics with JSON export."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self.start_time = datetime.now()
        
    def log_scalar(self, key: str, value: float, step: int):
        """Log a scalar metric."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append({"step": step, "value": value})
        
    def log_dict(self, metrics_dict: dict, step: int):
        """Log multiple metrics at once."""
        for key, value in metrics_dict.items():
            self.log_scalar(key, value, step)
            
    def save(self, filename: str = "metrics.json"):
        """Save metrics to JSON file."""
        output_path = self.log_dir / filename
        with open(output_path, 'w') as f:
            json.dump({
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'metrics': self.metrics
            }, f, indent=2)
        return output_path
EOF

# Create checkpointing module
cat > code/src/utils/checkpointing.py << 'EOF'
"""
Checkpointing utilities for model saving and loading.
"""

import torch
from pathlib import Path
from typing import Dict, Optional
import shutil
import json

class CheckpointManager:
    """Manage model checkpoints with versioning."""
    
    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        
    def save_checkpoint(
        self,
        agent,
        episode: int,
        metrics: Dict,
        filename: Optional[str] = None
    ) -> Path:
        """Save a checkpoint."""
        if filename is None:
            filename = f"checkpoint_ep{episode}.pth"
            
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save agent
        agent.save(str(checkpoint_path))
        
        # Save metadata
        metadata = {
            'episode': episode,
            'metrics': metrics
        }
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Manage old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
        
    def load_checkpoint(self, checkpoint_path: Path, agent):
        """Load a checkpoint."""
        agent.load(str(checkpoint_path))
        
        # Load metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata
        return {}
        
    def load_best_checkpoint(self, agent, metric: str = "mean_reward"):
        """Load the best checkpoint based on a metric."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
            
        best_checkpoint = None
        best_value = float('-inf')
        
        for cp in checkpoints:
            metadata_path = cp.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                value = metadata.get('metrics', {}).get(metric, float('-inf'))
                if value > best_value:
                    best_value = value
                    best_checkpoint = cp
                    
        if best_checkpoint:
            return self.load_checkpoint(best_checkpoint, agent)
        return {}
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_ep*.pth"),
            key=lambda p: p.stat().st_mtime
        )
        
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                checkpoint.unlink()
                metadata_path = checkpoint.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()
EOF

# Create config validation module
cat > code/src/utils/config.py << 'EOF'
"""
Configuration validation and management.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field, validator

class ChainConfig(BaseModel):
    name: str
    chain_id: int
    block_time: float = Field(gt=0)
    base_gas_price: float = Field(gt=0)
    gas_volatility: float = Field(ge=0, le=1)

class BridgeConfig(BaseModel):
    name: str
    source_chain: str
    target_chain: str
    capacity: float = Field(gt=0)
    latency_mean: float = Field(ge=0)
    latency_std: float = Field(ge=0)
    failure_rate: float = Field(ge=0, le=1)
    fee_basis_points: int = Field(ge=0, le=10000)

class PoolConfig(BaseModel):
    token_a: str
    token_b: str
    reserve_a: float = Field(gt=0)
    reserve_b: float = Field(gt=0)
    fee_tier: int = Field(ge=0)

class TrainingConfig(BaseModel):
    n_agents: int = Field(ge=1)
    hidden_dim: int = Field(ge=8)
    gamma: float = Field(ge=0, le=1)
    lr: float = Field(gt=0)
    n_episodes: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    device: str = "cpu"

def load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate chains
    for chain_config in config['environment']['chains']:
        ChainConfig(**chain_config)
    
    # Validate bridges
    for bridge_config in config['environment']['bridges']:
        BridgeConfig(**bridge_config)
    
    # Validate training config
    TrainingConfig(**config['training'])
    
    return config
EOF

# Create utils __init__.py
mkdir -p code/src/utils
cat > code/src/utils/__init__.py << 'EOF'
"""Utility modules for MARL training."""
from .logger import setup_logger, MetricsLogger
from .checkpointing import CheckpointManager
from .config import load_and_validate_config

__all__ = [
    'setup_logger',
    'MetricsLogger',
    'CheckpointManager',
    'load_and_validate_config'
]
EOF

echo "✓ Production features created successfully!"
