# Estimator Build and Install

## Setup

**CPU Base Docker Image**

| GCC Version | Python Version |                           IMAGE                           |
| ----------- | -------------- | --------------------------------------------------------- |
|   7.5.0     |    3.6.9       | alideeprec/deeprec-base:deeprec-base-cpu-py36-ubuntu18.04 |
|   9.4.0     |    3.8.10      | alideeprec/deeprec-base:deeprec-base-cpu-py38-ubuntu20.04 |
|   11.2.0    |    3.8.6       | alideeprec/deeprec-base:deeprec-base-cpu-py38-ubuntu22.04 |


**GPU Base Docker Image**

| GCC Version | Python Version | CUDA VERSION |                           IMAGE                                 |
| ----------- | -------------- | ------------ | --------------------------------------------------------------- |
|    7.5.0    |    3.6.9       | CUDA 11.6.1  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu116-ubuntu18.04 |
|    9.4.0    |    3.8.10      | CUDA 11.6.2  | alideeprec/deeprec-base:deeprec-base-gpu-py38-cu116-ubuntu20.04 |
|    11.2.0   |    3.8.6       | CUDA 11.7.1  | alideeprec/deeprec-base:deeprec-base-gpu-py38-cu117-ubuntu22.04 |


**CPU Dev Docker (with bazel cache)**

| GCC Version | Python Version |                           IMAGE                           |
| ----------- | -------------- | --------------------------------------------------------- |
|   7.5.0     |    3.6.9       | alideeprec/deeprec-build:deeprec-dev-cpu-py36-ubuntu18.04 |
|   9.4.0     |    3.8.10      | alideeprec/deeprec-build:deeprec-dev-cpu-py38-ubuntu20.04 |

**GPU(cuda11.6) Dev Docker (with bazel cache)**

| GCC Version | Python Version | CUDA VERSION |                           IMAGE                                 |
| ----------- | -------------- | ------------ | --------------------------------------------------------------- |
|    7.5.0    |    3.6.9       | CUDA 11.6.1  | alideeprec/deeprec-build:deeprec-dev-gpu-py36-cu116-ubuntu18.04 |
|    9.4.0    |    3.8.10      | CUDA 11.6.2  | alideeprec/deeprec-build:deeprec-dev-gpu-py38-cu116-ubuntu20.04 |

## Estimator Source Code

DeepRec provide new distributed protocols such as grpc++ and star_server, which is not supported when use native Estimator. We provide Estimator based on DeepRec.

Source Code: [https://github.com/DeepRec-AI/estimator](https://github.com/DeepRec-AI/estimator)

Develop Branch：master, Latest Release Branch: deeprec2402

## Estimator Build

**Build Package Builder**

```bash
bazel build //tensorflow_estimator/tools/pip_package:build_pip_package
```

**Build Package**

```bash
bazel-bin/tensorflow_estimator/tools/pip_package/build_pip_package /tmp/estimator_whl
```

## Estimator Install Package

Installing DeepRec will install the native tensorflow-estimator by default, please reinstall the compiled Estimator.
