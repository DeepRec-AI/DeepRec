# TFServing Build and Install

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


## TFServing Source Code

We provide optimized TFServing which could highly improve performance in inference, such as SessionGroup, CUDA multi-stream, etc.

Source Code: [https://github.com/DeepRec-AI/serving](https://github.com/DeepRec-AI/serving)

Develop Branch: master, Latest Release Branch: deeprec2306

## TFServing Build

**Build Package Builder-CPU**

```bash
bazel build -c opt tensorflow_serving/...
```

**Build CPU Package Builder with OneDNN + Eigen Threadpool**

```bash
bazel build  -c opt --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true tensorflow_serving/...
```

**Build Package Builder-GPU**

```bash
bazel build -c opt --config=cuda tensorflow_serving/...
```

**Build Package**

```bash
bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package /tmp/tf_serving_client_whl
```

**Server Bin**

Server Bin would generated in following directory:
```bash
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```
