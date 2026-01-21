# HIE-GAN: Hierarchical Implicit-Explicit GAN

This repository implements the HIE-GAN architecture for 3D reconstruction from 2D images. 

## 🚀 Quick Setup (Docker)

We strongly recommend using Docker to ensure a consistent environment with correct CUDA versions.

1.  **Build the Image**:
    ```bash
    docker build -t hiegan-dev .
    ```

2.  **Run the Container**:
    > **⚠️ Important**: You must use `--shm-size=8g` to prevent "Bus error" during data loading.
    ```bash
    ```bash
    docker run --gpus all --shm-size=8g -it \
        --name hiegan-container \
        -v $(pwd):/workspace \
        hiegan-dev
    ```

3.  **Resume Existing Container**:
    To avoid creating multiple containers, restart your existing one:
    ```bash
    docker start -ai hiegan-container
    ```

For detailed Docker troubleshooting, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

---

## 🛠️ Main Command Interface

All functionalities (Training, Inference, Plotting) are consolidated into a single entry point: `src/main.py`.

**Basic Syntax:**
```bash
python3 src/main.py --mode [train|resume|generate|plot] [arguments]
```

---

## 🏋️ Training

### 1. Start Fresh Training
```bash
python3 src/main.py --mode train --epochs 50 --batch-size 64 --exp-name my_experiment
```
*   **Outputs**: Saved to `output/DATE/TIME/` (contains `checkpoints/`, `logs/`).
*   **Automation**: By default, it will automatically plot graphs and generate sample 3D models at the end of training.

### 2. Resume Training
If a checkpoint exists in your experiment folder, `--mode train` will auto-resume. To **force** resume from a specific checkpoint:
```bash
python3 src/main.py \
  --mode resume \
  --checkpoint output/10-12-2025/07-06-29/checkpoints/checkpoint_latest.pth
```

### Key Training Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--epochs N` | Number of epochs to train | 50 (Config) |
| `--batch-size N` | Batch size | 64 (Config) |
| `--num-workers N` | DataLoader workers | 4 |
| `--exp-name NAME` | Name for the experiment folder | `default` |
| `--generate-graph` | Auto-plot loss graphs after training | `True` |
| `--generate-model` | Auto-generate samples after training | `True` |

---

## 🧪 Inference (Generation)

Generate 3D meshes (`.obj`) from images using a trained model.

### 1. Single Image Generation
Generates a 3D mesh for a specific input image. Saves **both** the generated `.obj` and the input `.png`.

```bash
python3 src/main.py \
  --mode generate \
  --image path/to/my_image.png \
  --checkpoint output/YOUR_EXP/checkpoints/checkpoint_best.pth
```
**Output Location:** `output/YOUR_EXP/generated_samples/`

### 2. Batch Generation (Dataset)
Generates `N` samples **per class** from the test dataset.

```bash
python3 src/main.py \
  --mode generate \
  --checkpoint output/YOUR_EXP/checkpoints/checkpoint_best.pth \
  --num-samples 20 \
  --exp-name batch_results
```
**Output Location:** `output/YOUR_EXP/batch_results/`
*   Creates subfolders for each class (e.g., `airplane/`, `chair/`).
*   `--num-samples 20` means it will generate 20 chairs, 20 airplanes, etc.

### 📉 Output Directory Rules
The generator is smart about where it saves files to keep your project organized:

1.  **Default**: `output/YOUR_EXP/generated_samples` (Sibling to `checkpoints` folder).
2.  **With `--exp-name subdir`**: `output/YOUR_EXP/subdir`.
3.  **With `--output-dir /abs/path`**: `/abs/path` (Complete override).

---

## 📊 Visualization (Plotting)

Manually plot training loss graphs from a log file.

```bash
python3 src/main.py \
  --mode plot \
  --log-dir output/YOUR_EXP/phase1_train.csv
```
> **Tip**: You can pass the `.log` file path, and the tool will automatically find the corresponding `.csv` file for you!

**Outputs**: `loss_vs_step.png` and `loss_vs_epoch.png` in the same folder as the log.

---

## 📂 Project Structure

```
HIE-GAN/
├── data/                       # ShapeNet Dataset
├── output/                     # Experiment Outputs
│   └── DD-MM-YYYY/
│       └── HH-MM-SS/
│           ├── checkpoints/    # Saved (.pth) models
│           ├── logs/           # Text and CSV logs
│           ├── graphs/         # Loss plots
│           └── generated/      # Inference outputs
├── src/
│   ├── configs/                # YAML Configs (train, model, dataset)
│   ├── dataloader/             # ShapeNet Dataset loader
│   ├── models/                 # Encoder & ExplicitDeformer
│   ├── trainer/                # Training Loop
│   ├── utils/                  # Checkpoints, Logger, Plotter
│   ├── main.py                 # 🚀 Unified Entry Point
│   └── generate.py             # Generation Logic
└── Dockerfile                  # Environment Definition
```
