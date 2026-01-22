import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path

def collect_merged_metrics(exp_dir):
    """
    Recursively find and merge metrics.json files.
    Returns a unified metrics dictionary.
    """
    exp_dir = Path(exp_dir)
    
    # 1. Try to find metrics.json directly
    metrics_file = exp_dir / "metrics.json"
    metrics_list = []
    
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics_list.append(json.load(f))
        except Exception as e:
            print(f"⚠️ Failed to read {metrics_file}: {e}")
            
    # 2. Look for subdirectories (Recursive Mode)
    # We look for all metrics.json files in subfolders
    # Exclude the root one if we already added it (to avoid duplication if glob matches it)
    sub_files = sorted(exp_dir.glob("**/metrics.json"))
    
    if not metrics_file.exists() and not sub_files:
        print(f"❌ No metrics.json found in {exp_dir} or its subdirectories.")
        return None

    # Identify files we haven't loaded yet
    for pf in sub_files:
        if pf.resolve() == metrics_file.resolve():
            continue
            
        try:
            with open(pf, 'r') as f:
                data = json.load(f)
                # rudimentary check to ensure it's a valid metrics file
                if "epochs" in data or "steps" in data:
                    metrics_list.append(data)
        except Exception as e:
            print(f"⚠️ Failed to read {pf}: {e}")
            
    if not metrics_list:
        return None
        
    print(f"✓ Found {len(metrics_list)} metrics files. Merging...")
    
    # Merge Metrics
    # We assume the files are sorted by Glob (alphanumeric path). 
    # Usually timestamps (YYYY-MM-DD/HH-MM-SS) sort correctly.
    merged_metrics = {"epochs": [], "steps": []}
    
    for m in metrics_list:
        if "epochs" in m:
            merged_metrics["epochs"].extend(m["epochs"])
        if "steps" in m:
            merged_metrics["steps"].extend(m["steps"])
            
    # Remove duplicate epochs if any? 
    # For now, we assume simple appending is what the user wants (concatenation of runs).
    
    return merged_metrics

def plot_training_graphs(exp_dir, output_dir=None, interval=20):
    """
    Generate comprehensive training graphs from metrics.json (or aggregated)
    
    Args:
        exp_dir: Experiment directory
        output_dir: Output directory for graphs
        interval: X-axis tick interval
    """
    exp_dir = Path(exp_dir)
    
    metrics = collect_merged_metrics(exp_dir)
    if not metrics:
        return False
    
    # Setup output directory
    if output_dir is None:
        output_dir = exp_dir / "graphs"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        epochs = metrics.get("epochs", [])
        
        if not epochs:
            print(f"⚠️  No epoch data found in metrics.")
            return False
        
        # Extract data
        epoch_nums = []
        train_losses = []
        val_losses = []
        test_losses = []
        
        # GAN Losses
        d_losses = []
        g_adv_losses = []
        
        for entry in epochs:
            if isinstance(entry, (int, str)) and entry == "test":
                continue
            
            if isinstance(entry, dict):
                epoch_num = entry.get("epoch")
                if epoch_num is None or epoch_num == "test":
                    continue
                    
                epoch_nums.append(epoch_num)
                train_losses.append(entry.get("train_loss"))
                val_losses.append(entry.get("val_loss"))
                
                # Test loss
                test_loss = entry.get("test_loss")
                if test_loss is not None:
                    test_losses.append(test_loss)
                    
                # GAN Losses
                if "d_loss" in entry:
                    d_losses.append(entry["d_loss"])
                if "g_adv" in entry:
                    g_adv_losses.append(entry["g_adv"])
        
        # Remove None values for standard losses
        valid_train = [(e, l) for e, l in zip(epoch_nums, train_losses) if l is not None]
        valid_val = [(e, l) for e, l in zip(epoch_nums, val_losses) if l is not None]
        
        train_epochs, train_loss_values = zip(*valid_train) if valid_train else ([], [])
        val_epochs, val_loss_values = zip(*valid_val) if valid_val else ([], [])
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Helper to set xticks
        def set_xticks(epochs_list):
            if not epochs_list: return
            start = int(min(epochs_list))
            end = int(max(epochs_list))
            # Ensure we have at least one tick
            ticks = list(range(start, end + 1, interval))
            if not ticks: ticks = [start]
            plt.xticks(ticks)

        # 1. Train Loss Plot
        if valid_train:
            plt.figure(figsize=(10, 6))
            plt.plot(train_epochs, train_loss_values, 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            set_xticks(train_epochs)
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
            set_xticks(val_epochs)
            plt.tight_layout()
            val_plot_path = output_dir / "val_loss.png"
            plt.savefig(val_plot_path, dpi=150)
            plt.close()
            print(f"✓ Saved: {val_plot_path}")
        
        # 3. Train vs Val Loss Comparison
        if valid_val and valid_train:
            plt.figure(figsize=(12, 6))
            # Use intersection of epochs for cleaner plot? Or just plot all.
            plt.plot(train_epochs, train_loss_values, 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss', alpha=0.8)
            plt.plot(val_epochs, val_loss_values, 'g-', linewidth=2, marker='s', markersize=4, label='Val Loss', alpha=0.8)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Train vs Validation Loss', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            set_xticks(list(train_epochs) + list(val_epochs))
            plt.tight_layout()
            comparison_plot_path = output_dir / "train_vs_val_loss.png"
            plt.savefig(comparison_plot_path, dpi=150)
            plt.close()
            print(f"✓ Saved: {comparison_plot_path}")
        
        # 4. GAN Loss Plot (if available)
        # We assume d_losses and g_adv_losses align with the first N epochs of train_epochs
        # Or more accurately, we should zip them with epoch_nums where they exist
        if len(d_losses) > 0 and len(g_adv_losses) > 0:
            # Re-align with epochs just to be safe
            gan_data = []
            for entry in epochs:
                if isinstance(entry, dict) and "d_loss" in entry and "g_adv" in entry:
                    gan_data.append((entry["epoch"], entry["d_loss"], entry["g_adv"]))
            
            if gan_data:
                g_epochs, g_d, g_g = zip(*gan_data)
                
                plt.figure(figsize=(10, 6))
                plt.plot(g_epochs, g_d, 'r-', linewidth=2, label='Discriminator Loss', alpha=0.7)
                plt.plot(g_epochs, g_g, 'b-', linewidth=2, label='Generator Adv Loss', alpha=0.7)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title('GAN Training Stability (D vs G)', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                set_xticks(g_epochs)
                plt.tight_layout()
                gan_plot_path = output_dir / "gan_loss_stability.png"
                plt.savefig(gan_plot_path, dpi=150)
                plt.close()
                print(f"✓ Saved: {gan_plot_path}")
        
        # 5. Test Loss (if available)
        if test_losses:
            avg_test_loss = sum(test_losses) / len(test_losses)
            plt.figure(figsize=(6, 6))
            plt.bar(['Test Loss'], [avg_test_loss], color='orange', alpha=0.7, width=0.5)
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
    parser.add_argument("exp_dir", type=str, help="Experiment directory (contains metrics.json or subfolders)")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for graphs")
    parser.add_argument("--interval", "-i", type=int, default=20, help="X-axis tick interval (default: 20)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy CSV plotting (provide CSV path as exp_dir)")
    
    args = parser.parse_args()
    
    if args.legacy:
        plot_training_logs(args.exp_dir, args.output_dir)
    else:
        plot_training_graphs(args.exp_dir, args.output_dir, args.interval)
