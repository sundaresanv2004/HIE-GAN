# HIE-GAN Command Reference

This guide provides a comprehensive list of commands and options for running the HIE-GAN project. The entry point for all operations is `src/main.py`.

## 🚀 Quick Start

### Train new model
```bash
python src/main.py --mode train --exp-name my_experiment
```

### Phase 3 Training (Fusion Module)
To train the full model with Explicit, Implicit, and Fusion branches:
```bash
python src/main.py --mode train --exp-name phase3_fusion
```

### Verification (Overfit Test)
Run a quick test on a few samples to verify gradient flow and pipeline integrity:
```bash
python src/main.py --mode train \
    --exp-name verification_test \
    --epochs 10 \
    --batch-size 2 \
    --num-samples 4 \
    --inspect-data
```

---

## 📊 Dataset Management

The HIE-GAN project includes powerful dataset validation tools accessible through `main.py`. These commands help you verify dataset integrity, check structure, and get detailed statistics.

### Dataset Structure Check

Validate your dataset structure and identify any missing files or folders.

#### Check All Splits
Checks train, val, and test directories automatically:
```bash
python src/main.py --mode dataset-check
```

**Output includes:**
- Total objects per class
- Valid objects (with required `images/` folder and `model_normalized.ply`)
- Missing components per class
- Overall dataset statistics

#### Check Specific Split
Validate only training data:
```bash
python src/main.py --mode dataset-check --split train
```

Validate only validation data:
```bash
python src/main.py --mode dataset-check --split val
```

Validate only test data:
```bash
python src/main.py --mode dataset-check --split test
```

#### Custom Dataset Path
Check dataset in a custom location:
```bash
python src/main.py --mode dataset-check --data-root /path/to/ShapeNetCore_V5
```

Check specific split in custom location:
```bash
python src/main.py --mode dataset-check \
    --data-root /content/data/ShapeNetCore_V5 \
    --split train
```

#### Detailed Statistics
Get class-by-class breakdown with `--detailed`:
```bash
python src/main.py --mode dataset-check --detailed
```

**Detailed output shows:**
- Per-class object counts
- Valid vs total objects ratio
- Split-wise distribution
- Missing files analysis

### Dataset Commands Reference

| Command | Description | Example |
| :--- | :--- | :--- |
| **Basic Check** | Check all splits with default path | `python src/main.py --mode dataset-check` |
| **Specific Split** | Check only one split (train/val/test) | `--mode dataset-check --split train` |
| **Custom Path** | Use custom dataset directory | `--mode dataset-check --data-root /path/to/data` |
| **Detailed Stats** | Show per-class breakdown | `--mode dataset-check --detailed` |
| **Combined** | Custom path + specific split + details | `--mode dataset-check --data-root /path --split val --detailed` |

### Common Use Cases

#### 1. Before Training - Verify Dataset
```bash
# Quick validation before starting training
python src/main.py --mode dataset-check

# Detailed check of training split
python src/main.py --mode dataset-check --split train --detailed
```

#### 2. After Dataset Download - Integrity Check
```bash
# After extracting dataset from tar.gz
python src/main.py --mode dataset-check \
    --data-root /content/data/ShapeNetCore_V5 \
    --detailed
```

#### 3. Google Colab - Fast Validation
```bash
# Check dataset on Colab storage
python src/main.py --mode dataset-check \
    --data-root /content/data/ShapeNetCore_V5 \
    --split train
```

#### 4. Debugging Data Loader Issues
```bash
# If training fails, check dataset structure first
python src/main.py --mode dataset-check --detailed

# Then run dry-run to test data loading
python src/main.py --mode train --dry-run --inspect-data
```

### Expected Dataset Structure

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
│   ├── 03636649/            # Lamp class (optional)
│   └── 04379243/            # Table class
├── val/                      # Validation split
│   └── ...
└── test/                     # Test split (optional)
    └── ...
```

### Interpreting Results

**✅ Valid Object**: Has both `images/` folder and `model_normalized.ply` file  
**⚠️ Missing Images**: Object folder exists but no `images/` subdirectory  
**⚠️ Missing PLY**: Object folder exists but no `model_normalized.ply` file

---

## 🛠 Training Commands

### Basic Training
Start a fresh training run using default configurations (`src/configs/*.yaml`).
```bash
python src/main.py --mode train
```

### Common Overrides
You can override config values directly from the CLI:

```bash
python src/main.py --mode train \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0002 \
    --num-workers 4
```

### Debug / Dry Run
Useful for verifying setup without waiting for training.
```bash
# Dry run: Initialize everything, check dataset, and exit before training loop
python src/main.py --mode train --dry-run

# Debug mode: Run with small data subset and verbose logging
python src/main.py --mode train --debug
```

### Checkpointing
- **Resume**: Automatically resumes if a checkpoint exists in the log dir, or specify one:
  ```bash
  python src/main.py --mode resume --checkpoint path/to/ckpt.pth
  ```
- **Scratch**: Force start from scratch even if checkpoints exist:
  ```bash
  python src/main.py --mode scratch
  ```
### Testing & Evaluation
- **Automatic Testing**: By default, evaluation on the test set is run after training completes. To disable this:
  ```bash
  python src/main.py --mode train --no-test
  ```
- **Manual Testing**: Run evaluation on a pre-trained model:
  ```bash
  python src/main.py --mode test --checkpoint output/my_exp/checkpoints/checkpoint_best.pth
  ```

---

## 🧪 Inference & Generation

### Single Image Inference
Generate explicit, implicit, and fused meshes from a single input image.
```bash
python src/main.py --mode generate \
    --checkpoint output/phase3_fusion/checkpoints/checkpoint_best.pth \
    --image assets/chair.png \
    --output-dir results/
```

### Batch Generation
Generate meshes for a random batch of validation images.
```bash
python src/main.py --mode generate \
    --checkpoint output/phase3_fusion/checkpoints/checkpoint_best.pth \
    --num-samples 10
```

---

## 📂 Output Structure
Experiments are organized by name, date, and time:
```
output/
  └── exp_name/
      └── DD-MM-YYYY/
          └── HH-MM-SS/
              ├── checkpoints/
              ├── logs/
              ├── metrics.json
              └── val_outputs/   <-- Generated mesh samples during training
```

---

## 📊 Visualization

### Plot Training Loss
Generate loss curves from a training CSV log.
```bash
python src/main.py --mode plot --log-dir output/phase3_fusion/DD-MM-YYYY/HH-MM-SS/phase1_train.csv
```

---

## ⚙️ CLI Options Reference

### Global
| Flag | Description |
| :--- | :--- |
| `--mode` | Execution mode: `train`, `resume`, `scratch`, `test`, `generate`, `plot`, `dataset-check` |
| `--config-dir` | Directory containing YAML configs (default: `src/configs`) |
| `--device` | Compute device: `auto`, `cuda`, `cpu`, `cuda:0` (default: `auto`) |

### Training Hyperparameters
| Flag | Description |
| :--- | :--- |
| `--epochs` | Number of training epochs |
| `--batch-size` | Batch size |
| `--lr` | Learning rate |
| `--weight-decay` | Optimizer weight decay |
| `--grad-clip` | Gradient clipping max norm |
| `--mixed-precision` | Enable FP16 training (saves memory) |
| `--compile` | Use `torch.compile` (PyTorch 2.0+ speedup) |

### Data Options
| Flag | Description |
| :--- | :--- |
| `--data-root` | Override dataset root directory |
| `--split` | Dataset split to check: `train`, `val`, `test`, `all` (for dataset-check mode) |
| `--num-samples` | Limit dataset size (for testing) |
| `--inspect-data` | Log sample data stats during loading |
| `--detailed` | Show detailed dataset statistics (for dataset-check mode) |

### Logging & Checkpoints
| Flag | Description |
| :--- | :--- |
| `--exp-name` | Name of the experiment (folder name in `output/`) |
| `--save-every` | Save checkpoint every N epochs |
| `--keep-last` | Number of recent checkpoints to keep |
| `--val-every` | Run validation every N epochs |
| `--quiet` | Minimal console output |
| `--no-tqdm` | Disable progress bars |
