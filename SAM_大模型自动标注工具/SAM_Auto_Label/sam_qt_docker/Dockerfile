FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive

# ===== 系统层依赖（Qt xcb 必需）=====
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

# ===== 创建 conda 环境 =====
RUN conda create -n samqt python=3.10 -y
SHELL ["conda", "run", "-n", "samqt", "/bin/bash", "-c"]

# ===== Qt + PyQt + OpenCV（关键）=====
RUN conda install -y -c conda-forge \
    pyqt=5 \
    qt=5 \
    opencv \
    numpy \
    onnxruntime

# ===== 工作目录 =====
WORKDIR /workspace

