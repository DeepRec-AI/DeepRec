# DeepRec Build and Install

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
|    7.5.0    |    3.6.9       | CUDA 11.0.3  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu110-ubuntu18.04 |
|    7.5.0    |    3.6.9       | CUDA 11.2.2  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu112-ubuntu18.04 |
|    7.5.0    |    3.6.9       | CUDA 11.4.2  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu114-ubuntu18.04 |
|    7.5.0    |    3.6.9       | CUDA 11.6.1  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu116-ubuntu18.04 |
|    7.5.0    |    3.6.9       | CUDA 11.7.1  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu117-ubuntu18.04 |
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


## Build

**GPU Environment**

Configure TF_CUDA_COMPUTE_CAPABILITIES could improve performance, please follow to setup correct TF_CUDA_COMPUTE_CAPABILITIES.

| GPU architecture    | TF_CUDA_COMPUTE_CAPABILITIES |
| ------------------- | ---------------------------- |
| Pascal (P100)       | 6.0+6.1                      |
| Volta (V100)        | 7.0                          |
| Turing (T4)         | 7.5                          |
| Ampere (A10, A100)  | 8.0+8.6                      |

If you need to compile DeepRec wheel that run on different GPU architecture, configure TF_CUDA_COMPUTE_CAPABILITIES such as:


```bash
export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0"
```

**Configuration**
```bash
./configure
```

**Build GPU/CPU Package Builder**

```bash
bazel build -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
```

**Build GPU/CPU Package Builder with ABI=0**

```bash
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
```

**Build CPU Package Builder with OneDNN + Eigen Threadpool**

```bash
bazel build  -c opt --config=opt  --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true //tensorflow/tools/pip_package:build_pip_package
```

**Build CPU Package Builder with OneDNN + Eigen Threadpool + ABI=0**

```bash
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --config=opt --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true //tensorflow/tools/pip_package:build_pip_package
```

## Build Package

```bash
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

## Install Package

```bash
pip3 install /tmp/tensorflow_pkg/tensorflow-1.15.5+${version}-cp38-cp38m-linux_x86_64.whl
```

## Latest Release Images

**CPU Image**

```
alideeprec/deeprec-release:deeprec2302-cpu-py38-ubuntu20.04
```

**GPU Image with CUDA 11.6**

```
alideeprec/deeprec-release:deeprec2302-gpu-py38-cu116-ubuntu20.04
```
