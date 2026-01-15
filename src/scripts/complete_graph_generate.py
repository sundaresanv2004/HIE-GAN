import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration - Update these paths
BASE_DIR = "/Users/sundar/Downloads"  # Change this to your directory
CSV_FILES = [
    "phase1_train_1.csv",
    "phase1_train_2.csv",
    "phase1_train_3.csv",
    "phase1_train_4.csv",
    "phase1_train_5.csv",
    "phase1_train_6.csv",
    "phase1_train_7.csv",
]

# Plotting configuration
EPOCH_SKIP = 0  # 0 = plot all epochs, 1 = skip 1 epoch (every 2nd), 2 = skip 2 epochs (every 3rd), etc.

# Read and combine all CSV files
dataframes = []
for csv_file in CSV_FILES:
    file_path = os.path.join(BASE_DIR, csv_file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Combine all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Sort by timestamp to ensure chronological order
combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

# Get the last step of each epoch
epoch_data = combined_df.groupby('epoch').last().reset_index()

# Apply epoch skip filter
if EPOCH_SKIP > 0:
    # Plot every (EPOCH_SKIP + 1)th epoch
    plot_data = epoch_data[epoch_data.index % (EPOCH_SKIP + 1) == 0].copy()
    print(f"Plotting every {EPOCH_SKIP + 1} epoch(s): {len(plot_data)} points out of {len(epoch_data)} total epochs")
else:
    plot_data = epoch_data.copy()
    print(f"Plotting all epochs: {len(plot_data)} points")

# Create the loss plot
plt.figure(figsize=(12, 6))
plt.plot(plot_data['epoch'], plot_data['loss'], marker='o', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot

output_path = os.path.join(BASE_DIR, 'training_loss_plot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Display the plot
plt.show()

# Print summary statistics (using all epoch data)
print(f"\nTraining Summary:")
print(f"Total epochs: {epoch_data['epoch'].max()}")
print(f"Starting loss: {epoch_data['loss'].iloc[0]:.4f}")
print(f"Final loss: {epoch_data['loss'].iloc[-1]:.4f}")
print(f"Loss reduction: {epoch_data['loss'].iloc[0] - epoch_data['loss'].iloc[-1]:.4f}")
print(f"\nFirst few epochs:")
print(epoch_data.head(10))
print(f"\nLast few epochs:")
print(epoch_data.tail(10))
