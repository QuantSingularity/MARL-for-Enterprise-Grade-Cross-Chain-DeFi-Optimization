"""Checkpointing utilities for model saving and loading."""

import json
from pathlib import Path
from typing import Dict, Optional


class CheckpointManager:
    """Manage model checkpoints with versioning."""

    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save_checkpoint(
        self, agent, episode: int, metrics: Dict, filename: Optional[str] = None
    ) -> Path:
        """Save a checkpoint."""
        if filename is None:
            filename = f"checkpoint_ep{episode}.pth"
        checkpoint_path = self.checkpoint_dir / filename
        agent.save(str(checkpoint_path))
        metadata = {"episode": episode, "metrics": metrics}
        with open(checkpoint_path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)
        self._cleanup_old_checkpoints()
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path, agent):
        """Load a checkpoint."""
        agent.load(str(checkpoint_path))
        metadata_path = checkpoint_path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_ep*.pth"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[: -self.keep_last_n]:
                checkpoint.unlink()
                metadata_path = checkpoint.with_suffix(".json")
                if metadata_path.exists():
                    metadata_path.unlink()
