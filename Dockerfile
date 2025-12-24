# Base image with CUDA 12.1 and cuDNN 8
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

# Set timezone and locale
ENV TZ=Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6 python3-pip python3.8 python3.8-venv \
    && rm -rf /var/lib/apt/lists/*

# Upgrade security packages
RUN apt upgrade --no-install-recommends -y openssl tar

# Set Python version
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install additional Python packages (example)
RUN pip install numpy pandas matplotlib scikit-learn

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Default command
CMD ["bash"]
