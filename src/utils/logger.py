import logging
import os
import csv
from datetime import datetime


def setup_logger(log_dir, filename):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger("HIEGAN")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # file handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    logger.info(f"Logging at: {log_path}")
    return logger


class CSVLogger:
    def __init__(self, log_dir, filename):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self._init()

    def _init(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "epoch", "step", "loss"])

    def write(self, epoch, step, loss):
        ts = datetime.now().isoformat()
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, epoch, step, loss])
