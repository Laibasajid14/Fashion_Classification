"""
utils/logger.py
Structured logging to console + JSON log file for both models.
"""

import os
import sys
import json
import logging
from datetime import datetime


def setup_logger(name, log_dir, filename="training.log"):
    """
    Set up a logger that writes to both stdout and a file.

    Args:
        name:    logger name (e.g. 'baseline', 'improved')
        log_dir: directory where log file is saved
        filename: log filename

    Returns:
        logging.Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logger initialised → {log_path}")
    return logger


class EpochLogger:
    """
    Accumulates per-epoch metrics and saves to a JSON log file.
    """

    def __init__(self, log_dir, filename="epoch_log.json"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.records = []

    def log(self, epoch, **kwargs):
        record = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **kwargs}
        self.records.append(record)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

    def get_history(self):
        """Return dict of lists for plotting."""
        if not self.records:
            return {}
        keys = [k for k in self.records[0].keys() if k not in ("epoch", "timestamp")]
        return {k: [r.get(k) for r in self.records] for k in keys}
