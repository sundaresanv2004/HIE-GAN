import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path


def plot_training_graphs(exp_dir, output_dir=None):
    """
    Generate comprehensive training graphs from metrics.json
    
    Creates:
    - train_loss.png: Training loss over epochs
    - val_loss.png: Validation loss over epochs  
    - train_vs_val_loss.png: Train and Val loss comparison
    - test_loss.png: Test loss (if available)
    
    Args:
        exp_dir: Experiment directory containing metrics.json
        output_dir: Output directory for graphs (default: exp_dir/graphs)
    """
    exp_dir = Path(exp_dir)
    metrics_file = exp_dir / "metrics.json"
    
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return False
    
    # Setup output directory
    if output_dir is None:
        output_dir = exp_dir / "graphs"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        epochs = metrics.get("epochs", [])
        
        if not epochs:
            print(f"⚠️  No epoch data found in metrics.json")
            return False
        
        # Extract data
        epoch_nums = []
        train_losses = []
        val_losses = []
        test_losses = []
        
        for entry in epochs:
            if isinstance(entry, (int, str)) and entry == "test":
                # Test entry
                continue
            
            if isinstance(entry, dict):
                epoch_num = entry.get("epoch")
                if epoch_num is None or epoch_num == "test":
                    continue
                    
                epoch_nums.append(epoch_num)
                train_losses.append(entry.get("train_loss"))
                val_losses.append(entry.get("val_loss"))
                
                # Test loss might be in a separate entry
                test_loss = entry.get("test_loss")
                if test_loss is not None:
                    test_losses.append(test_loss)
        
        # Remove None values
        valid_train = [(e, l) for e, l in zip(epoch_nums, train_losses) if l is not None]
        valid_val = [(e, l) for e, l in zip(epoch_nums, val_losses) if l is not None]
        
        if not valid_train:
            print("⚠️  No valid training loss data found")
            return False
        
        train_epochs, train_loss_values = zip(*valid_train) if valid_train else ([], [])
        val_epochs, val_loss_values = zip(*valid_val) if valid_val else ([], [])
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Train Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_loss_values, 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        train_plot_path = output_dir / "train_loss.png"
        plt.savefig(train_plot_path, dpi=150)
        plt.close()
        print(f"✓ Saved: {train_plot_path}")
        
        # 2. Val Loss Plot (if available)
        if valid_val:
            plt.figure(figsize=(10, 6))
            plt.plot(val_epochs, val_loss_values, 'g-', linewidth=2, marker='s', markersize=4, label='Validation Loss')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Validation Loss vs Epoch', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            val_plot_path = output_dir / "val_loss.png"
            plt.savefig(val_plot_path, dpi=150)
            plt.close()
            print(f"✓ Saved: {val_plot_path}")
        
        #3. Train vs Val Loss Comparison (if val available)
        if valid_val:
            plt.figure(figsize=(12, 6))
            plt.plot(train_epochs, train_loss_values, 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss', alpha=0.8)
            plt.plot(val_epochs, val_loss_values, 'g-', linewidth=2, marker='s', markersize=4, label='Val Loss', alpha=0.8)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Train vs Validation Loss', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            comparison_plot_path = output_dir / "train_vs_val_loss.png"
            plt.savefig(comparison_plot_path, dpi=150)
            plt.close()
            print(f"✓ Saved: {comparison_plot_path}")
        
        # 4. Test Loss (if available)
        # Check if there's a test entry in metrics
        test_entry = None
        for entry in epochs:
            if isinstance(entry, dict) and entry.get("epoch") == "test":
                test_entry = entry
                break
        
        if test_entry and test_entry.get("test_loss") is not None:
            test_loss_value = test_entry["test_loss"]
            
            # Bar chart for test loss
            plt.figure(figsize=(6, 6))
            plt.bar(['Test Loss'], [test_loss_value], color='orange', alpha=0.7, width=0.5)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Test Loss', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            test_plot_path = output_dir / "test_loss.png"
            plt.savefig(test_plot_path, dpi=150)
            plt.close()
            print(f"✓ Saved: {test_plot_path}")
        
        print(f"\n✅ All graphs saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate graphs: {e}")
        import traceback
        traceback.print_exc()
        return False


def plot_training_logs(csv_path, output_dir=None):
    """
    LEGACY: Reads the training CSV log and plots Loss vs Step/Epoch.
    Kept for backward compatibility.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ Log file not found: {csv_path}")
        return

    if output_dir is None:
        output_dir = csv_path.parent / "graphs"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        
        # Plot Loss vs Step
        plt.figure(figsize=(10, 6))
        plt.plot(df['step'], df['loss'], label='Training Loss', alpha=0.6)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Step')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "loss_vs_step.png")
        plt.close()

        # Aggregate by Epoch if available
        if 'epoch' in df.columns:
            epoch_df = df.groupby('epoch')['loss'].mean().reset_index()
            plt.figure(figsize=(10, 6))
            plt.plot(epoch_df['epoch'], epoch_df['loss'], marker='o', label='Avg Epoch Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Average Training Loss vs Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / "loss_vs_epoch.png")
            plt.close()
            
        print(f"✓ Graphs saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Failed to plot logs: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training graphs")
    parser.add_argument("exp_dir", type=str, help="Experiment directory (contains metrics.json)")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for graphs")
    parser.add_argument("--legacy", action="store_true", help="Use legacy CSV plotting (provide CSV path as exp_dir)")
    
    args = parser.parse_args()
    
    if args.legacy:
        plot_training_logs(args.exp_dir, args.output_dir)
    else:
        plot_training_graphs(args.exp_dir, args.output_dir)
