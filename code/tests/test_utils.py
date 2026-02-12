"""Tests for utility modules."""

from code.src.utils.logger import MetricsLogger


def test_metrics_logger(tmp_path):
    """Test metrics logger."""
    logger = MetricsLogger(tmp_path)
    logger.log_scalar("reward", 1.0, 0)
    logger.log_scalar("reward", 2.0, 1)
    logger.log_dict({"loss": 0.5, "accuracy": 0.9}, 0)
    output_file = logger.save()
    assert output_file.exists()
