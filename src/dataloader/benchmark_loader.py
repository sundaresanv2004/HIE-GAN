import time
import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloader.dataset import DatasetLoader, ShapeNetDataset
from utils.config import load_configs

class MockArgs:
    def __init__(self):
        self.config_dir = "src/configs"
        self.debug = False # Ensure we load full dataset if possible or at least enough to test
        self.num_samples = None 
        self.val_split = 0.0
        self.seed = 42
        self.pin_memory = True
        self.inspect_data = False

class MockLogger:
    def info(self, msg): print(msg)
    def warning(self, msg): print(f"WARNING: {msg}")

def benchmark():
    args = MockArgs()
    logger = MockLogger()
    
    # Load default config
    dataset_cfg, _, train_cfg = load_configs(args.config_dir)
    
    # Override for benchmark
    train_cfg["batch_size"] = 16
    train_cfg["num_workers"] = 4
    
    loader = DatasetLoader(dataset_cfg, train_cfg, args, logger)
    train_loader, _ = loader.load(ShapeNetDataset)
    
    if train_loader is None or len(train_loader) == 0:
        print("Dataset is empty. Cannot benchmark.")
        return

    print("\n🚀 Starting Benchmark...")
    print(f"Batch Size: {train_cfg['batch_size']}")
    print(f"Workers: {train_cfg['num_workers']}")
    
    start_time = time.time()
    count = 0
    num_batches = 50 # Limit to 50 batches to save time
    
    for i, batch in enumerate(train_loader):
        count += 1
        if i >= num_batches:
            break
        
        if i % 10 == 0:
            print(f"Loaded batch {i}...")

    end_time = time.time()
    duration = end_time - start_time
    total_samples = count * train_cfg['batch_size']
    
    print(f"\n⏱️  Total Time: {duration:.2f}s")
    print(f"📦 Total Batches: {count}")
    print(f"🖼️  Total Samples: {total_samples}")
    print(f"⚡ Throughput: {total_samples / duration:.2f} samples/sec")

if __name__ == "__main__":
    benchmark()
