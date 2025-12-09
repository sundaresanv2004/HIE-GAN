import logging
import os
import csv
import json
from datetime import datetime
from pathlib import Path


def setup_logger(log_dir, filename, quiet=False):
    """Setup dual logger (console + file)"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    logger = logging.getLogger("HIEGAN")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    if not quiet:
        from tqdm import tqdm
        
        class TqdmLoggingHandler(logging.Handler):
            def __init__(self, level=logging.NOTSET):
                super().__init__(level)

            def emit(self, record):
                try:
                    msg = self.format(record)
                    tqdm.write(msg)
                    self.flush()
                except Exception:
                    self.handleError(record)

        ch = TqdmLoggingHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    if not quiet:
        logger.info(f"Logging initialized: {log_path}")

    return logger


class CSVLogger:
    """CSV logger for training metrics"""

    def __init__(self, log_dir, filename):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / filename
        self._init_file()

    def _init_file(self):
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "epoch", "step", "loss"])

    def write(self, epoch, step, loss):
        ts = datetime.now().isoformat()
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, epoch, step, f"{loss:.8f}"])


class MetricsLogger:
    """JSON-based metrics logger"""

    def __init__(self, log_dir, filename):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / filename
        self.metrics = {"epochs": [], "steps": []}
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r") as f:
                self.metrics = json.load(f)

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def log_epoch(self, epoch, data):
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **data}
        self.metrics["epochs"].append(entry)
        self._save()

    def log_step(self, epoch, step, data):
        entry = {"epoch": epoch, "step": step, "timestamp": datetime.now().isoformat(), **data}
        self.metrics["steps"].append(entry)
        self._save()
