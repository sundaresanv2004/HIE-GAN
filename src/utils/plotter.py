import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_training_logs(csv_path, output_dir=None):
    """
    Reads the training CSV log and plots Loss vs Step/Epoch.
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
    parser = argparse.ArgumentParser(description="Plot training logs")
    parser.add_argument("csv_path", type=str, help="Path to training_log.csv")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for graphs")
    
    args = parser.parse_args()
    plot_training_logs(args.csv_path, args.output_dir)
