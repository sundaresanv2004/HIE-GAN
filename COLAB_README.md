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
Run this to connect your Drive and install all libraries (`trimesh`, `rtree`, `pytorch-geometric`).
```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install System Deps (Required for Rtree/Trimesh)
!apt-get install -y libspatialindex-dev

# 3. Install Python Dependencies
print("⏳ Installing dependencies... (2-3 mins)")
!pip install -r colab_requirements.txt

# 4. Install PyTorch Geometric (Correct CUDA version)
import torch
cuda_version = torch.version.cuda.replace(".", "")
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cu{cuda_version}.html

# 5. Verify Setup
print("✅ Verifying setup...")
!python src/main.py --mode train --dry-run
```

### Step 3: Unpack Dataset (Fast I/O)
Reading from Drive is slow. We extract `ShapeNetCore_V5.tar.gz` to local Colab disk.
```python
# 1. Create local data folder
!mkdir -p /content/data

# 2. Extract .tar.gz from Drive -> Local
# REPLACE with accurate path if different
tar_path = "/content/drive/MyDrive/ShapeNetCore_V5.tar.gz"

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

### Step 5: Inference & Visualization
Generate meshes using the trained model.
```python
ckpt_path = f"{output_path}/phase3_colab_run/checkpoints/checkpoint_best.pth"

!python src/main.py --mode generate \
    --checkpoint $ckpt_path \
    --num-samples 5 \
    --output-root $output_path \
    --data-root $data_path
```
