# HIE-GAN Command Reference

This guide provides a comprehensive list of commands and options for running the HIE-GAN project. The entry point for all operations is `src/main.py`.

## 🚀 Quick Start

### Train new model
```bash
python src/main.py --mode train --exp-name my_experiment
```

### Resume training
```bash
python src/main.py --mode train --resume-from output/my_experiment/checkpoints/checkpoint_latest.pth
```

### Generate 3D Meshes from Images
```bash
python src/main.py --mode generate \
    --checkpoint output/my_experiment/checkpoints/checkpoint_best.pth \
    --image assets/chair.png
```

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
# Dry run: Initialize everything but exit before training loop
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

---

## 🧪 Inference & Generation

### Single Image Inference
Generate a 3D mesh from a single input image.
```bash
python src/main.py --mode generate \
    --checkpoint <path_to_checkpoint> \
    --image <path_to_image> \
    --output-dir results/
```

### Batch Generation
Generate meshes for a random batch of validation images.
```bash
python src/main.py --mode generate \
    --checkpoint <path_to_checkpoint> \
    --num-samples 10
```

---

## 📊 Visualization

### Plot Training Loss
Generate loss curves from a training CSV log.
```bash
python src/main.py --mode plot --log-dir output/my_experiment/phase1_train.csv
```

---

## ⚙️ CLI Options Reference

### Global
| Flag | Description |
| :--- | :--- |
| `--mode` | Execution mode: `train`, `resume`, `scratch`, `generate`, `plot`. |
| `--config-dir` | Directory containing YAML configs (default: `src/configs`). |
| `--device` | Compute device: `auto`, `cuda`, `cpu`, `cuda:0` (default: `auto`). |

### Training Hyperparameters
| Flag | Description |
| :--- | :--- |
| `--epochs` | Number of training epochs. |
| `--batch-size` | Batch size. |
| `--lr` | Learning rate. |
| `--weight-decay` | Optimizer weight decay. |
| `--grad-clip` | Gradient clipping max norm. |
| `--mixed-precision` | Enable FP16 training (saves memory). |
| `--compile` | Use `torch.compile` (PyTorch 2.0+ speedup). |

### Data
| Flag | Description |
| :--- | :--- |
| `--data-root` | Override dataset root directory. |
| `--num-samples` | Limit dataset size (for testing). |
| `--inspect-data` | Log sample data stats during loading. |

### Logging & Checkpoints
| Flag | Description |
| :--- | :--- |
| `--exp-name` | Name of the experiment (folder name in `output/`). |
| `--save-every` | Save checkpoint every N epochs. |
| `--keep-last` | Number of recent checkpoints to keep. |
| `--val-every` | Run validation every N epochs. |
| `--quiet` | Minimal console output. |
| `--no-tqdm` | Disable progress bars. |
