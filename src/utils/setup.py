import os
import random
import numpy as np
import torch
import datetime
from pathlib import Path

class EnvironmentSetup:
    """Helper class for setting up the training environment."""
    
    @staticmethod
    def set_seed(seed, deterministic=False):
        """
        Sets random seeds for reproducibility.
        
        Args:
            seed (int): The random seed.
            deterministic (bool): Whether to enforce deterministic algorithms (slower).
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def setup_directories(args, train_cfg):
        """
        Creates experiment and checkpoint directories based on timestamp and config.

        Args:
            args: Parsed command line arguments.
            train_cfg (dict): Training configuration dictionary.

        Returns:
            tuple: (exp_dir (Path), ckpt_dir (Path))
        """
        # Get current timestamp
        now = datetime.datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H-%M-%S")
        
        # Base output directory
        if args.exp_name:
            base_dir = Path("output") / args.exp_name / date_str / time_str
        else:
            # Fallback if no experiment name provided
            base_dir = Path("output") / "default" / date_str / time_str

        # Update log dir in config so other components know where to log
        train_cfg["logging"]["log_dir"] = str(base_dir)
        train_cfg["checkpoints"]["dir"] = str(base_dir / "checkpoints")

        exp_dir = base_dir
        ckpt_dir = base_dir / "checkpoints"

        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        return exp_dir, ckpt_dir

    @staticmethod
    def setup_device(device_str, quiet=False):
        """
        Configures the compute device.

        Args:
            device_str (str): Device string from args (e.g., 'auto', 'cuda').
            quiet (bool): Suppress print statements.

        Returns:
            torch.device: The selected device.
        """
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                if not quiet:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"✓ Auto-selected GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                device = torch.device("cpu")
                if not quiet:
                    print("⚠ No GPU available, using CPU")
        else:
            device = torch.device(device_str)
            if not quiet:
                if device.type == "cuda":
                    gpu_name = torch.cuda.get_device_name(device)
                    print(f"✓ Using GPU: {gpu_name}")
                else:
                    print(f"✓ Using device: {device}")

        return device
