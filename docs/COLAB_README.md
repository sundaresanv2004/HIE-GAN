# ☁️ Running HIE-GAN on Google Colab

Follow these steps exactly to run your project on Colab.

### Step 1: Clone Repository
Copy this into the first cell to get your code (Phase 2/3 branch):
```python
import os
repo_url = "https://github.com/sundaresanv2004/HIE-GAN.git" 
branch_name = "phase-2-implicit"

if not os.path.exists("HIE-GAN"):
    !git clone -b $branch_name $repo_url
else:
    print("Project already cloned!")

%cd HIE-GAN
print(f"📂 Working directory: {os.getcwd()}")
```

### Step 2: Mount Drive & Install Environment
Run this to connect your Drive and install libraries.
**Note**: We strictly force **PyTorch 2.4.0** + **CUDA 12.4** as per your reference.

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install System Deps
!apt-get install -y libspatialindex-dev

# 3. Force PyTorch 2.4.0 (The working version from your screenshot)
print("🔄 Installing PyTorch 2.4.0...")
!pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install PyTorch Geometric (Binary Wheels for 2.4.0)
print("⬇️ Installing PyTorch Geometric...")
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# 5. Install other dependencies
print("⏳ Installing remaining dependencies...")
!pip install -r colab_requirements.txt

# 6. Verify Setup
import torch
print(f"✅ Setup Complete. PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
!python src/main.py --mode train --dry-run
```

### Step 3: Unpack Dataset (Fast I/O)
Reading from Drive is slow. We extract `ShapeNetCore_V5.tar.gz` from Drive to local Colab disk.
```python
# 1. Create local data folder
!mkdir -p /content/data

# 2. Extract .tar.gz from Drive -> Local
# REPLACE with accurate path if different
tar_path = "/content/drive/MyDrive/Dataset/ShapeNetCore_V5.tar.gz"

if os.path.exists(tar_path):
    print(f"⏳ Extracting {tar_path}...")
    !tar -xzf $tar_path -C /content/data
    print("✅ Extraction complete!")
    !ls /content/data/ShapeNetCore_V5
else:
    print(f"❌ File not found: {tar_path}")
```

### Step 4: Run Training (Phase 3)
This saves checkpoints directly to your Drive (`--output-root`).

#### Option A: Train with Automatic Testing (Default)
Test evaluation runs automatically after training completes.
```python
# Define paths
data_path = "/content/data/ShapeNetCore_V5"
output_path = "/content/drive/MyDrive/HIE-GAN/output"

!python src/main.py --mode train \
    --exp-name phase3_colab_run \
    --data-root $data_path \
    --output-root $output_path \
    --batch-size 64 \
    --epochs 20
```

#### Option B: Train WITHOUT Testing (Faster)
Use `--no-test` to skip automatic test evaluation after training.
```python
# Define paths
data_path = "/content/data/ShapeNetCore_V5"
output_path = "/content/drive/MyDrive/HIE-GAN/output"

!python src/main.py --mode train \
    --exp-name phase3_colab_run \
    --data-root $data_path \
    --output-root $output_path \
    --batch-size 64 \
    --epochs 20 \
    --no-test
```

> **Note**: You can manually run testing later using the command in Step 5a below.

### Step 5a: Manual Test Evaluation (Optional)
If you trained with `--no-test`, you can manually run test evaluation:
```python
# Define paths
data_path = "/content/data/ShapeNetCore_V5"
output_path = "/content/drive/MyDrive/HIE-GAN/output"
ckpt_path = f"{output_path}/phase3_colab_run/checkpoints/checkpoint_best.pth"

!python src/main.py --mode test \
    --checkpoint $ckpt_path \
    --data-root $data_path \
    --output-root $output_path
```

### Step 5b: Generate Meshes (Inference)
Generate meshes using the trained model.
```python
ckpt_path = f"{output_path}/phase3_colab_run/checkpoints/checkpoint_best.pth"

!python src/main.py --mode generate \
    --checkpoint $ckpt_path \
    --num-samples 5 \
    --output-root $output_path \
    --data-root $data_path
```

---

## Troubleshooting

### ❌ Error: "No objects found in dataset" or "Train objects: 0"

**Problem:** The dataset loader can't find your data files.

**Solution:** Check your data structure matches one of these formats:

#### Format 1: With explicit train/val splits (Recommended)
```
ShapeNetCore_V5/
├── train/
│   ├── 02691156/          # Airplane
│   │   ├── <object_id>/
│   │   │   ├── images/
│   │   │   │   ├── 00.png
│   │   │   │   └── ... (multiple images)
│   │   │   └── model_normalized.ply
│   │   └── ...
│   ├── 02958343/          # Car
│   └── ...
├── val/
│   ├── 02691156/
│   └── ...
└── test/                  # Optional (use --no-test if missing)
    └── ...
```

#### Format 2: Without splits (auto-split)
```
ShapeNetCore_V5/
├── 02691156/              # Airplane
│   ├── <object_id>/
│   │   ├── images/
│   │   │   ├── 00.png
│   │   │   └── ... (multiple images)
│   │   └── model_normalized.ply
│   └── ...
├── 02958343/              # Car
└── ...
```

**Verify your data:**
```python
import os
data_path = "/content/data/ShapeNetCore_V5"

# Check if path exists
if os.path.exists(data_path):
    print(f"✓ Data path exists: {data_path}")
    print(f"Contents: {os.listdir(data_path)}")
else:
    print(f"❌ Data path NOT found: {data_path}")

# Check for a sample object
sample_class = "02691156"  # Airplane
class_path = f"{data_path}/{sample_class}"
if os.path.exists(class_path):
    print(f"✓ Class directory exists")
    objects = os.listdir(class_path)
    print(f"Found {len(objects)} objects")
    if objects:
        sample_obj = f"{class_path}/{objects[0]}"
        print(f"Sample object contents: {os.listdir(sample_obj)}")
else:
    print(f"❌ Class directory NOT found: {class_path}")
```

### ⚠️ Limited Storage: Skipping Test Set

If you're skipping the test directory to save space:
1. **Always use `--no-test`** flag when training
2. The code will now work fine without a test directory
3. You can still validate your model using the validation set

### 🔍 Check Configuration

Verify your `configs/dataset.yaml` or command-line path:
```python
# If data is in: /content/data/ShapeNetCore_V5
# Your command should use:
!python src/main.py --data-root /content/data/ShapeNetCore_V5 ...

# NOT:
# !python src/main.py --data-root /content/data ...  ❌ Wrong!
```
