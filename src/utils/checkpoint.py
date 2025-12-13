import torch
import json
import logging
from pathlib import Path
from datetime import datetime

class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.
    
    Attributes:
        ckpt_dir (Path): Directory to store checkpoints.
        logger (logging.Logger): Logger instance.
        keep_last (int): Number of recent checkpoints to allow conservation of.
    """

    def __init__(self, ckpt_dir, logger, keep_last=3):
        self.ckpt_dir = Path(ckpt_dir)
        self.logger = logger
        self.keep_last = keep_last
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state, epoch, loss, is_best=False, best_loss=None):
        """
        Saves a checkpoint.

        Args:
            state (dict): State dictionary containing model, optimizer, etc.
            epoch (int): Current epoch number.
            loss (float): Current validation or training loss.
            is_best (bool): Whether this is the best checkpoint so far.
            best_loss (float, optional): The best loss value tracked.
        """
        # Save epoch checkpoint
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(state, ckpt_path)

        # Save latest
        latest_path = self.ckpt_dir / "checkpoint_latest.pth"
        torch.save(state, latest_path)

        # Save best
        if is_best:
            best_path = self.ckpt_dir / "checkpoint_best.pth"
            torch.save(state, best_path)
            self.logger.info(f"  💾 Best checkpoint saved (loss: {loss:.6f})")

        # Save metadata
        self._save_metadata(epoch, loss, best_loss, is_best)

        # Rotate old checkpoints
        self._rotate_checkpoints()

    def load(self, path, device):
        """
        Loads a checkpoint from the specified path.

        Args:
            path (str or Path): Path to the checkpoint file.
            device (torch.device): Device to map the checkpoint to.

        Returns:
            dict: The loaded checkpoint dictionary.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        self.logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)
        return checkpoint

    def _save_metadata(self, epoch, loss, best_loss, is_best):
        """Saves training metadata to a JSON file."""
        metadata = {
            "last_epoch": epoch,
            "best_epoch": -1, 
            "last_loss": loss,
            "best_loss": best_loss,
            "timestamp": datetime.now().isoformat()
        }

        meta_path = self.ckpt_dir.parent / "training_metadata.json"
        
        # Preserve best_epoch if it exists
        if meta_path.exists() and not is_best:
            try:
                with open(meta_path, 'r') as f:
                    old_meta = json.load(f)
                    metadata["best_epoch"] = old_meta.get("best_epoch", -1)
            except Exception:
                pass
        
        if is_best:
            metadata["best_epoch"] = epoch

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def _rotate_checkpoints(self):
        """Removes old checkpoints to save disk space."""
        all_checkpoints = sorted(self.ckpt_dir.glob("checkpoint_epoch_*.pth"))
        if len(all_checkpoints) > self.keep_last:
            for old_ckpt in all_checkpoints[:-self.keep_last]:
                old_ckpt.unlink()
                # Use debug info instead of typical info to reduce noise, unless important
                # The original code logged this at info level
                # self.logger.info(f"  🗑️  Removed old checkpoint: {old_ckpt.name}") 
