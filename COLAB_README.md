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
Test mode is enabled by default to evaluate performance after training.
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

### Step 5: Reuse & Inference
Generate meshes using the trained model.
```python
ckpt_path = f"{output_path}/phase3_colab_run/checkpoints/checkpoint_best.pth"

!python src/main.py --mode generate \
    --checkpoint $ckpt_path \
    --num-samples 5 \
    --output-root $output_path \
    --data-root $data_path
```
