# syntax=docker/dockerfile:1

#---------------------HOMAI------------------------#
# Created on Sun Mar 31 2024
#
# Copyright (c) 2024 The Home Made AI (HOMAI)
# Author: Javad Rezaie
# License: Apache License 2.0
#---------------------HOMAI------------------------#


ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine, MMCV, MMDetection and MMYolo
RUN pip install openmim && \
    mim install "mmengine>=0.6.0" "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0,<4.0.0"

# Install JupyterLab if you are interested to run experiments on the docker and seaborn to plot logs
RUN mim install jupyterlab ipykernel seaborn
RUN git clone https://github.com/open-mmlab/mmyolo.git &&\
    cd mmyolo && pip install albumentations==1.3.1 &&\
    mim install -v -e .
RUN pip install sahi
RUN git clone https://github.com/open-mmlab/mmdetection.git