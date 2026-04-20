# Move to CUDA 11.1 to support the RTX 3060
FROM nvcr.io/nvidia/cuda:11.1.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3.8 as default and install specific Pip for 3.8
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && rm get-pip.py

WORKDIR /workspaces/condlanenet