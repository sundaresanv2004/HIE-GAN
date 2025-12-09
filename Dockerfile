# BASE IMAGE WITH CUDA + PYTORCH
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# SYSTEM SETUP
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    python3-opengl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# INSTALL uv PACKAGE MANAGER
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# PROJECT WORKDIR
WORKDIR /workspace

# DEPENDENCIES
COPY requirements.txt /workspace/
RUN uv pip install --system -r requirements.txt

# INSTALL torch-geometric with ALL dependencies (CUDA 12.1)
RUN uv pip install --system torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html && \
    uv pip install --system torch-geometric

# COPY PROJECT FILES
COPY . /workspace

# DEVELOPMENT ENTRYPOINT
ENTRYPOINT ["/bin/bash"]
