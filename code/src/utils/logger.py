"""
Logging utilities for MARL training and evaluation.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with file and console handlers."""

    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

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
        with open(output_path, "w") as f:
            json.dump(
                {
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "metrics": self.metrics,
                },
                f,
                indent=2,
            )
        return output_path
