# DOCKER SETUP GUIDE

## Prerequisites (All Systems)

- Install Docker Desktop
- Clone the repository so you have:

```
/HIE-GAN
    Dockerfile
    requirements.txt
    ...
```

---

# Linux GPU Setup (CUDA Machines)

## Verify NVIDIA Driver

```
nvidia-smi
```

If this prints GPU info → OK  
If not, install NVIDIA drivers.

## Install NVIDIA Container Toolkit

Follow:  
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Restart Docker after install:

```
sudo systemctl restart docker
```

## Build Image

```
docker build -t hiegan-dev .
```

## Run GPU Container

```
docker run --gpus all --shm-size=8g -it \
    --name hiegan-container \
    -v $(pwd):/workspace \
    hiegan-dev
```

## Reuse Exiting Container (Don't create new ones)

If you have already created a container, you can restart it instead of creating a new one (which wastes disk space):

```bash
docker start -ai hiegan-container
```

---

# Windows CUDA Setup (WSL2 Required)

> Windows native Docker cannot access CUDA directly.
> You must use WSL2 + Ubuntu.

## Requirements

- Windows 10 or 11
- WSL2 enabled
- Ubuntu installed under WSL2
- NVIDIA GPU drivers installed for WSL2

## Test GPU inside WSL

```
nvidia-smi
```

If this prints GPU info → OK

## Install NVIDIA Container Toolkit inside WSL Ubuntu

Use the same official guide:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Build and Run in WSL Ubuntu

```
docker build -t hiegan-dev .
```

Run with GPU:

```
docker run --gpus all --shm-size=8g -it \
    --name hiegan-container \
    -v $(pwd):/workspace \
    hiegan-dev
```

### Resume Existing Container

```bash
docker start -ai hiegan-container
```

---

# Verify CUDA Availability (Inside Container)

Inside container shell, run:

```
python3 - <<EOF
import torch
print("CUDA available? ->", torch.cuda.is_available())
print("GPU name ->", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF
```

## Expected Output

### macOS or CPU-only environment:
```
CUDA available? -> False
GPU name -> None
```

### Linux GPU or Windows WSL GPU:
```
CUDA available? -> True
GPU name -> NVIDIA RTX XXXXX
```

---

# Notes

- On macOS: CUDA will always be False (expected)
- On GPU Linux or Windows WSL: CUDA should be True
- Always mount workspace for development:

```
docker run ... -v $(pwd):/workspace
```

This allows:
- Editing files live from host
- Creating and deleting files
- No rebuilds for code changes

---

# Platform Summary

| Platform | Build Image | GPU Training | Works |
|---|---|---|---|
| macOS | Yes | No | Yes for development/testing |
| Linux GPU | Yes | Yes | Best option |
| Windows WSL GPU | Yes | Yes | Works if toolkit installed |

