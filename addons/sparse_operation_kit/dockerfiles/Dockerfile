FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV HOROVOD_GPU_OPERATIONS=NCCL

RUN apt-get update && \
    apt-get install -y --allow-unauthenticated \
    wget \
    cmake \
    git \
    unzip \
    curl \
    libssl-dev \
    libcurl4-openssl-dev \
    zlib1g-dev \
    python3 \
    python3-dev \
    python3-pip \
    libopenmpi-dev \
    libboost-serialization-dev \
    && apt-get clean && \
    ln -sf python3 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip

RUN pip install \
    astor==0.8.1 \
    numpy==1.16.6 \
    scikit-build \
    twine \
    cmake==3.21.1 \
    protobuf==3.17.3 && \
    pip install --no-deps \
    keras-preprocessing==1.0.5

RUN wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh && \
    bash bazel-0.26.1-installer-linux-x86_64.sh
