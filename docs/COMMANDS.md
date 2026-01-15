# HIE-GAN Command Reference Guide

**Complete reference for all HIE-GAN commands and operations**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Commands](#training-commands)
3. [Dataset Management](#dataset-management)
4. [Inference & Generation](#inference--generation)
5. [Visualization](#visualization)
6. [Resume & Testing](#resume--testing)
7. [CLI Options Reference](#cli-options-reference)
8. [Output Structure](#output-structure)
9. [Common Workflows](#common-workflows)

---

## Quick Start

### 🚀 Train a New Model
```bash
python src/main.py --mode train --exp-name my_experiment --epochs 50
```

### 🔄 Resume Training
```bash
# Option 1: Provide output directory (auto-finds best checkpoint)
python src/main.py --mode resume --checkpoint output/my_experiment/16-01-2026/01-23-45

# Option 2: Provide specific checkpoint file
python src/main.py --mode resume --checkpoint output/my_experiment/.../checkpoints/checkpoint_best.pth
```

### 🧪 Run Test Evaluation
```bash
python src/main.py --mode test --checkpoint output/my_experiment/16-01-2026/01-23-45
```

### 🎨 Generate Meshes
```bash
python src/main.py --mode generate --checkpoint output/my_experiment/16-01-2026/01-23-45 --num-samples 10
```

---

## Training Commands

### Basic Training

Start a fresh training run with default configurations:
```bash
python src/main.py --mode train --exp-name phase3_fusion
```

**What happens**:
- Creates experiment directory: `output/phase3_fusion/DD-MM-YYYY/HH-MM-SS/`
- Trains with configurations from `src/configs/*.yaml`
- Saves checkpoints automatically
- Runs validation every epoch (by default)
- Generates plots and runs test evaluation after training

### Training with Custom Parameters

Override configuration values from command line:
```bash
python src/main.py --mode train \
    --exp-name custom_run \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0002 \
    --num-workers 8
```

**Available Overrides**:
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--weight-decay`: Optimizer weight decay
- `--num-workers`: Data loader workers
- `--grad-clip`: Gradient clipping threshold

### Feature Toggles

Disable specific features during training for faster iteration:

**Disable Validation**:
```bash
python src/main.py --mode train --no-validation --exp-name quick_debug
```
✅ Skips validation, trains faster

**Disable Plot Generation**:
```bash
python src/main.py --mode train --no-plot --exp-name no_plots
```
✅ Skips automatic graph generation after training

**Disable Test Evaluation**:
```bash
python src/main.py --mode train --no-test --exp-name no_testing
```
✅ Skips test set evaluation after training

**Combined Toggles**:
```bash
python src/main.py --mode train --no-validation --no-plot --no-test --exp-name minimal_run
```
✅ Train-only mode (fastest for debugging)

### Debug Mode

Run quick verification without full training:

**Dry Run** (Initialize only, no training):
```bash
python src/main.py --mode train --dry-run
```
✅ Checks dataset, builds models, initializes optimizer, then exits

**Debug Mode** (Small subset):
```bash
python src/main.py --mode train --debug --num-samples 100
```
✅ Trains on limited data for quick testing

**Inspect Data**:
```bash
python src/main.py --mode train --inspect-data --num-samples 10
```
✅ Logs detailed data statistics during loading

### Custom Paths

**Custom Dataset Path**:
```bash
python src/main.py --mode train --data-root /path/to/ShapeNetCore_V5
```

**Custom Output Directory**:
```bash
python src/main.py --mode train --output-root /path/to/my_experiments
```

**Custom Config Directory**:
```bash
python src/main.py --mode train --config-dir /path/to/custom_configs
```

---

## Dataset Management

Verify dataset integrity and get detailed statistics.

### Check All Splits

Automatically detects and validates train, val, and test directories:
```bash
python src/main.py --mode dataset-check
```

**Output**:
- Total objects per class
- Valid objects (with required files)
- Missing components
- Overall statistics

### Check Specific Split

**Training Data Only**:
```bash
python src/main.py --mode dataset-check --split train
```

**Validation Data**:
```bash
python src/main.py --mode dataset-check --split val
```

**Test Data**:
```bash
python src/main.py --mode dataset-check --split test
```

### Custom Dataset Path

Check dataset in different location:
```bash
python src/main.py --mode dataset-check --data-root /content/data/ShapeNetCore_V5
```

**With Specific Split**:
```bash
python src/main.py --mode dataset-check \
    --data-root /path/to/dataset \
    --split train
```

### Detailed Statistics

Get per-class breakdown:
```bash
python src/main.py --mode dataset-check --detailed
```

**Output includes**:
- Per-class object counts
- Valid vs total objects ratio
- Split-wise distribution
- Missing files analysis

### Dataset Commands Reference Table

| Command | Description | Example |
|:--------|:------------|:--------|
| **Basic Check** | Validate all splits | `--mode dataset-check` |
| **Specific Split** | Check one split | `--mode dataset-check --split train` |
| **Custom Path** | Use custom directory | `--mode dataset-check --data-root /path` |
| **Detailed Stats** | Show per-class info | `--mode dataset-check --detailed` |
| **Combined** | All options together | `--mode dataset-check --data-root /path --split train --detailed` |

### Common Dataset Use Cases

**1. Before Training - Quick Validation**:
```bash
python src/main.py --mode dataset-check
```

**2. After Download - Integrity Check**:
```bash
python src/main.py --mode dataset-check --data-root /content/data/ShapeNetCore_V5 --detailed
```

**3. Google Colab - Fast Check**:
```bash
python src/main.py --mode dataset-check --data-root /content/data/ShapeNetCore_V5 --split train
```

**4. Debug Data Issues**:
```bash
# First check dataset
python src/main.py --mode dataset-check --detailed

# Then dry-run training
python src/main.py --mode train --dry-run --inspect-data
```

---

## Resume & Testing

### Resume Training

HIE-GAN supports **smart checkpoint resuming** with two flexible methods.

#### Method 1: Output Directory (Recommended)

Simply provide the experiment output directory:
```bash
python src/main.py --mode resume --checkpoint output/phase3/16-01-2026/01-23-45
```

**Smart Detection**:
- 🏆 Automatically loads `checkpoint_best.pth` (highest priority)
- ⏰ Falls back to `checkpoint_latest.pth` (if no best)
- 📝 Uses most recent `checkpoint_epoch_*.pth` (fallback)

#### Method 2: Direct Checkpoint File

Specify exact checkpoint:
```bash
python src/main.py --mode resume \
    --checkpoint output/phase3/16-01-2026/01-23-45/checkpoints/checkpoint_best.pth
```

#### Resume with Overrides

Continue training with modified parameters:
```bash
python src/main.py --mode resume \
    --checkpoint output/phase3/16-01-2026/01-23-45 \
    --epochs 150 \
    --lr 5e-5 \
    --batch-size 64
```

#### Alternative Resume Methods

**From Checkpoints Directory**:
```bash
python src/main.py --mode resume \
    --checkpoint output/phase3/16-01-2026/01-23-45/checkpoints
```

**Auto-Resume** (uses latest in experiment dir):
```bash
python src/main.py --mode resume
```

### Test Evaluation

Run comprehensive evaluation on test set.

#### Test with Output Directory

```bash
python src/main.py --mode test --checkpoint output/phase3/16-01-2026/01-23-45
```
✅ Automatically uses best checkpoint

#### Test with Specific Checkpoint

```bash
python src/main.py --mode test \
    --checkpoint output/phase3/16-01-2026/01-23-45/checkpoints/checkpoint_epoch_0050.pth
```

**What Happens**:
- Loads model from checkpoint
- Evaluates on test set
- Calculates test loss
- Saves sample outputs to `test_outputs/`

---

## Inference & Generation

Generate 3D meshes from trained models.

### Single Image Generation

Generate mesh from a single input image:
```bash
python src/main.py --mode generate \
    --checkpoint output/phase3/16-01-2026/01-23-45 \
    --image assets/chair.png
```

**Output**:
- Explicit mesh: `..._explicit.obj`
- Implicit mesh: `..._implicit.obj`
- Fused mesh: `..._fused.obj`

### Batch Generation

Generate meshes for multiple random samples:
```bash
python src/main.py --mode generate \
    --checkpoint output/phase3/16-01-2026/01-23-45 \
    --num-samples 20
```

**Options**:
- `--num-samples`: Number of samples to generate (default: 20)

### Custom Output Directory

```bash
python src/main.py --mode generate \
    --checkpoint output/phase3/16-01-2026/01-23-45 \
    --output-dir results/my_generation \
    --num-samples 10
```

### Generation with Direct Checkpoint

```bash
python src/main.py --mode generate \
    --checkpoint output/phase3/.../checkpoints/checkpoint_best.pth \
    --num-samples 10
```

---

## Visualization

### Plot Training Logs

Generate loss curves from training logs.

#### Plot with Output Directory

```bash
python src/main.py --mode plot --log-dir output/phase3/16-01-2026/01-23-45
```
✅ Automatically finds and plots CSV log

#### Plot with CSV File

```bash
python src/main.py --mode plot --log-dir output/phase3/16-01-2026/01-23-45/training.csv
```

#### Plot with Log File (Auto-converts)

```bash
python src/main.py --mode plot --log-dir output/phase3/16-01-2026/01-23-45/training.log
```
ℹ️ Automatically converts `.log` → `.csv`

**What's Generated**:
- Training loss curve
- Validation loss curve
- Combined comparison plot
- Saved as PNG images

---

## CLI Options Reference

### Global Options

| Flag | Type | Description | Example |
|:-----|:-----|:------------|:--------|
| `--mode` | choice | Execution mode | `train`, `resume`, `test`, `generate`, `plot`, `dataset-check` |
| `--config-dir` | path | Config directory | `src/configs` |
| `--device` | string | Compute device | `auto`, `cuda`, `cpu`, `cuda:0` |
| `--seed` | int | Random seed | `42` |

### Training Parameters

| Flag | Type | Default | Description |
|:-----|:-----|:--------|:------------|
| `--exp-name` | string | `default` | Experiment name |
| `--epochs` | int | from config | Number of epochs |
| `--batch-size` | int | from config | Batch size |
| `--lr` | float | from config | Learning rate |
| `--weight-decay` | float | `0.0` | Weight decay |
| `--num-workers` | int | from config | Data loader workers |
| `--grad-clip` | float | None | Gradient clipping |

### Data Options

| Flag | Type | Description |
|:-----|:-----|:------------|
| `--data-root` | path | Override dataset root |
| `--output-root` | path | Base output directory |
| `--split` | choice | Dataset split: `train`, `val`, `test`, `all` |
| `--num-samples` | int | Limit dataset size |
| `--inspect-data` | flag | Log data statistics |
| `--detailed` | flag | Show detailed stats (dataset-check) |

### Feature Toggles

| Flag | Description |
|:-----|:------------|
| `--no-validation` | Disable validation during training |
| `--no-plot` | Disable automatic graph generation |
| `--no-test` | Skip test evaluation after training |
| `--no-save` | Disable checkpoint saving |
| `--no-tqdm` | Disable progress bars |

### Checkpoint & Logging

| Flag | Type | Description |
|:-----|:-----|:------------|
| `--checkpoint` | path | Checkpoint file or directory |
| `--log-dir` | path | Log file or directory |
| `--save-every` | int | Save checkpoint every N epochs |
| `--keep-last` | int | Keep last N checkpoints |
| `--val-every` | int | Validate every N epochs |
| `--quiet` | flag | Minimal console output |

### Debug & Development

| Flag | Description |
|:-----|:------------|
| `--dry-run` | Initialize only, no training |
| `--debug` | Enable debug mode |
| `--deterministic` | Enforce deterministic algorithms |
| `--mixed-precision` | Enable FP16 training |
| `--compile` | Use torch.compile (PyTorch 2.0+) |

### Generation Options

| Flag | Type | Description |
|:-----|:-----|:------------|
| `--image` | path | Single image for generation |
| `--num-samples` | int | Number of samples to generate |
| `--output-dir` | path | Output directory for generated meshes |

---

## Output Structure

All experiments are organized with date and timestamp subdirectories:

```
output/
└── <exp_name>/               # e.g., "phase3_fusion"
    └── DD-MM-YYYY/          # e.g., "16-01-2026"
        └── HH-MM-SS/        # e.g., "14-30-00"
            ├── checkpoints/
            │   ├── checkpoint_best.pth
            │   ├── checkpoint_latest.pth
            │   └── checkpoint_epoch_0020.pth
            ├── training.log
            ├── training.csv
            ├── training_metadata.json
            ├── metrics.json
            ├── val_outputs/
            │   └── epoch_XX/
            │       ├── sample_0_explicit.obj
            │       ├── sample_0_fused.obj
            │       └── sample_0_implicit.obj
            ├── test_outputs/
            │   └── test_XX_fused.obj
            └── generated_samples/
                ├── sample_0.obj
                └── sample_1.obj
```

**Key Features**:
- **Date Organization**: All runs on the same day grouped together
- **Timestamp Separation**: Each run has unique timestamp
- **Best Checkpoint**: Always saved as `checkpoint_best.pth`
- **Latest Checkpoint**: Updated with each save
- **Automatic Cleanup**: Old epoch checkpoints auto-deleted (keeps last 3)

---

## Common Workflows

### 1. Full Training Pipeline

```bash
# 1. Check dataset
python src/main.py --mode dataset-check --detailed

# 2. Train model
python src/main.py --mode train --exp-name production_v1 --epochs 100

# 3. Evaluate (happens automatically, or run manually)
python src/main.py --mode test --checkpoint output/production_v1/16-01-2026/14-30-00

# 4. Generate samples
python src/main.py --mode generate \
    --checkpoint output/production_v1/16-01-2026/14-30-00 \
    --num-samples 50
```

### 2. Fast Iteration Development

```bash
# Quick debug run (no validation, no plots, small dataset)
python src/main.py --mode train \
    --exp-name debug_run \
    --epochs 5 \
    --batch-size 8 \
    --num-samples 100 \
    --no-validation \
    --no-plot
```

### 3. Resume and Continue

```bash
# Initial training
python src/main.py --mode train --exp-name long_run --epochs 50

# Resume with more epochs
python src/main.py --mode resume \
    --checkpoint output/long_run/16-01-2026/14-30-00 \
    --epochs 150
```

### 4. Experiment with Different Parameters

```bash
# Baseline
python src/main.py --mode train --exp-name baseline --lr 1e-4

# Higher learning rate
python src/main.py --mode train --exp-name high_lr --lr 5e-4

# Different batch size
python src/main.py --mode train --exp-name large_batch --batch-size 128
```

### 5. Google Colab Workflow

```bash
# 1. Check dataset after upload
python src/main.py --mode dataset-check \
    --data-root /content/data/ShapeNetCore_V5 \
    --detailed

# 2. Train with Colab paths
python src/main.py --mode train \
    --exp-name colab_run \
    --data-root /content/data/ShapeNetCore_V5 \
    --output-root /content/drive/MyDrive/HIE-GAN/output

# 3. Generate plots
python src/main.py --mode plot \
    --log-dir /content/drive/MyDrive/HIE-GAN/output/colab_run/16-01-2026/14-30-00
```

### 6. Production Training

```bash
# Full production training with all features
python src/main.py --mode train \
    --exp-name production_model_v2 \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-4 \
    --mixed-precision \
    --save-every 5 \
    --val-every 2
```

---

## Expected Dataset Structure

The dataset checker validates this structure:

```
ShapeNetCore_V5/
├── train/                    # Training split
│   ├── 02691156/            # Airplane class
│   │   ├── <object_id>/
│   │   │   ├── images/      # ✅ Required
│   │   │   │   ├── 00.png
│   │   │   │   ├── 03.png
│   │   │   │   └── ...
│   │   │   └── model_normalized.ply  # ✅ Required
│   ├── 02958343/            # Car class
│   ├── 03001627/            # Chair class
│   ├── 03636649/            # Lamp class
│   └── 04379243/            # Table class
├── val/                      # Validation split
│   └── ... (same structure)
└── test/                     # Test split (optional)
    └── ... (same structure)
```

**Validation Results**:
- ✅ **Valid Object**: Has both `images/` folder and `model_normalized.ply`
- ⚠️ **Missing Images**: Object folder exists but no `images/` subdirectory
- ⚠️ **Missing PLY**: Object folder exists but no `model_normalized.ply`

---

## Tips & Best Practices

### Performance Optimization

1. **Use Mixed Precision**:
   ```bash
   --mixed-precision  # Faster training, lower memory
   ```

2. **Optimize Data Loading**:
   ```bash
   --num-workers 8  # Increase for faster data loading
   ```

3. **Gradient Clipping**:
   ```bash
   --grad-clip 1.0  # Prevent gradient explosions
   ```

### Experiment Organization

1. **Use descriptive experiment names**:
   ```bash
   --exp-name phase3_fusion_lr1e4_bs64
   ```

2. **Keep experiments organized by date automatically**

3. **Use `--output-root` for different projects**

### Checkpoint Management

1. **Always use output directory path** (recommended):
   ```bash
   --checkpoint output/exp/DATE/TIME
   ```

2. **System auto-selects best checkpoint**

3. **Adjust `--keep-last` to save disk space**:
   ```bash
   --keep-last 3  # Keep only 3 recent checkpoints
   ```

### Development Workflow

1. **Start with dry-run**:
   ```bash
   --dry-run  # Verify setup before training
   ```

2. **Use small dataset for debugging**:
   ```bash
   --num-samples 100 --no-validation --no-plot
   ```

3. **Check dataset before training**:
   ```bash
   --mode dataset-check --detailed
   ```

---

## Troubleshooting

### Common Issues

**Dataset Not Found**:
```bash
# Check dataset location
python src/main.py --mode dataset-check --data-root /path/to/dataset
```

**CUDA Out of Memory**:
```bash
# Reduce batch size
--batch-size 16

# Enable mixed precision
--mixed-precision
```

**Checkpoint Not Loading**:
```bash
# Verify checkpoint exists
ls output/exp/DATE/TIME/checkpoints/

# Try direct file path
--checkpoint output/exp/DATE/TIME/checkpoints/checkpoint_best.pth
```

**Training Too Slow**:
```bash
# Disable validation temporarily
--no-validation

# Increase workers
--num-workers 8

# Use mixed precision
--mixed-precision
```

---

## Additional Resources

- **Architecture Documentation**: See `docs/ARCHITECTURE.txt`
- **Colab Setup**: See `docs/COLAB_README.md`
- **Docker Setup**: See `docs/DOCKER_SETUP.md`
- **Main README**: See `README.md`

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Project**: HIE-GAN (Hierarchical Implicit-Explicit GAN)
