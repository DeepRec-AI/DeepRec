
![DeepRec Logo](https://github.com/alibaba/DeepRec/blob/main/docs/deeprec_logo.png)

--------------------------------------------------------------------------------

## **Introduction**
DeepRec is a high-performance recommendation deep learning framework based on [TensorFlow 1.15](https://www.tensorflow.org/), [Intel-TensorFlow](https://github.com/Intel-tensorflow/tensorflow) and [NVIDIA-TensorFlow](https://github.com/NVIDIA/tensorflow). It is hosted in incubation in LF AI & Data Foundation.


### **Background**
Recommendation models have huge commercial values for areas such as retailing, media, advertisements, social networks and search engines. Unlike other kinds of models, recommendation models have large amount of non-numeric features such as id, tag, text and so on which lead to huge parameters.

DeepRec has been developed since 2016, which supports core businesses such as Taobao Search, recommendation and advertising. It precipitates a list of features on basic frameworks and has excellent performance in recommendation models training and inference. So far, in addition to Alibaba Group, dozens of companies have used DeepRec in their business scenarios.

### **Key Features**
DeepRec has super large-scale distributed training capability, supporting recommendation model training of trillion samples and over ten trillion parameters. For recommendation models, in-depth performance optimization has been conducted across CPU and GPU platform. It contains list of features to improve usability and performance for super-scale scenarios. 

#### **Embedding & Optimizer**
 - Embedding Variable.
 - Dynamic Dimension Embedding Variable.
 - Adaptive Embedding Variable.
 - Multiple Hash Embedding Variable.
 - Multi-tier Hybrid Embedding Storage.
 - Group Embedding.
 - AdamAsync Optimizer.
 - AdagradDecay Optimizer.

#### **Training**
 - Asynchronous Distributed Training Framework (Parameter Server), such as grpc+seastar, FuseRecv, StarServer etc.
 - Synchronous Distributed Training Framework (Collective), such as HybridBackend, Sparse Operation Kits (SOK) etc.
 - Runtime Optimization, such as Graph Aware Memory Allocator (GAMMA), Critical-path based Executor etc.
 - Runtime Optimization (GPU), GPU Multi-Stream Engine which support multiple CUDA compute stream and CUDA Graph.
 - Operator level optimization, such as BF16 mixed precision optimization, embedding operator optimization and EmbeddingVariable on PMEM and GPU, new hardware feature enabling, etc.
 - Graph level optimization, such as AutoGraphFusion, SmartStage, AutoPipeline, Graph Template Engine, Sample-awared Graph Compression, MicroBatch etc.
 - Compilation optimization, support BladeDISC, XLA etc.

#### **Deploy and Serving**
 - Delta checkpoint loading and exporting.
 - Super-scale recommendation model distributed serving.
 - Multi-tier hybrid storage and multi backend supported.
 - Online deep learning with low latency.
 - High performance inference framework SessionGroup (share-nothing), with multiple threadpool and multiple CUDA stream supported.
 - Model Quantization.

***
## **Installation**

### **Prepare for installation**

**CPU Platform**

``````
alideeprec/deeprec-build:deeprec-dev-cpu-py38-ubuntu20.04
``````

**GPU Platform**

```
alideeprec/deeprec-build:deeprec-dev-gpu-py38-cu116-ubuntu20.04
```

### **How to Build**

Configure
```
$ ./configure
```
Compile for CPU and GPU defaultly
```
$ bazel build -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
```
Compile for CPU and GPU: ABI=0
```
$ bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
```
Compile for CPU optimization: oneDNN + Unified Eigen Thread pool
```
$ bazel build -c opt --config=opt --config=mkl_threadpool //tensorflow/tools/pip_package:build_pip_package
```
Compile for CPU optimization and ABI=0
```
$ bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --config=opt --config=mkl_threadpool //tensorflow/tools/pip_package:build_pip_package
```
### **Create whl package** 
```
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```
### **Install whl package**
```
$ pip3 install /tmp/tensorflow_pkg/tensorflow-1.15.5+${version}-cp38-cp38m-linux_x86_64.whl
```

### **Latest Release Images**

#### Image for CPU

```
alideeprec/deeprec-release:deeprec2310-cpu-py38-ubuntu20.04
```

#### Image for GPU CUDA11.6

```
alideeprec/deeprec-release:deeprec2310-gpu-py38-cu116-ubuntu20.04
```

***
## Continuous Build Status

### Official Build

| Build Type    | Status                                                       |
| ------------- | ------------------------------------------------------------ |
| **Linux CPU** | ![CPU Build](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-build-wheel.yaml/badge.svg) |
| **Linux GPU** | ![GPU Build](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-build-wheel.yaml/badge.svg) |
| **Linux CPU Serving** | ![CPU Serving Build](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-build-serving.yaml/badge.svg) |
| **Linux GPU Serving** | ![GPU Serving Build](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-build-serving-gpu.yaml/badge.svg) |

### Official Unit Tests

| Unit Test Type | Status |
| -------------- | ------ |
| **Linux CPU C** | ![CPU C Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-c-unit-test.yaml/badge.svg) |
| **Linux CPU CC** | ![CPU CC Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-cc-unit-test.yaml/badge.svg) |
| **Linux CPU Contrib** | ![CPU Contrib Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-contrib-unit-test.yaml/badge.svg) |
| **Linux CPU Core** | ![CPU Core Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-core-unit-test.yaml/badge.svg) |
| **Linux CPU Examples** | ![CPU Examples Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-examples-unit-test.yaml/badge.svg) |
| **Linux CPU Java** | ![CPU Java Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-java-unit-test.yaml/badge.svg) |
| **Linux CPU JS** | ![CPU JS Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-js-unit-test.yaml/badge.svg) |
| **Linux CPU Python** | ![CPU Python Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-python-unit-test.yaml/badge.svg) |
| **Linux CPU Stream Executor** | ![CPU Stream Executor Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-stream_executor-unit-test.yaml/badge.svg) |
| **Linux GPU C** | ![GPU C Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-c-unit-test.yaml/badge.svg) |
| **Linux GPU CC** | ![GPU CC Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-cc-unit-test.yaml/badge.svg) |
| **Linux GPU Contrib** | ![GPU Contrib Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-contrib-unit-test.yaml/badge.svg) |
| **Linux GPU Core** | ![GPU Core Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-core-unit-test.yaml/badge.svg) |
| **Linux GPU Examples** | ![GPU Examples Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-examples-unit-test.yaml/badge.svg) |
| **Linux GPU Java** | ![GPU Java Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-java-unit-test.yaml/badge.svg) |
| **Linux GPU JS** | ![GPU JS Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-js-unit-test.yaml/badge.svg) |
| **Linux GPU Python** | ![GPU Python Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-python-unit-test.yaml/badge.svg) |
| **Linux GPU Stream Executor** | ![GPU Stream Executor Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-stream_executor-unit-test.yaml/badge.svg) |
| **Linux CPU Serving UT** | ![CPU Serving Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-serving-unit-test.yaml/badge.svg) |
| **Linux GPU Serving UT** | ![GPU Serving Unit Tests](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-serving-unit-test-gpu.yaml/badge.svg) |

## **User Document**

Chinese: [https://deeprec.readthedocs.io/zh/latest/](https://deeprec.readthedocs.io/zh/latest/)

English: [https://deeprec.readthedocs.io/en/latest/](https://deeprec.readthedocs.io/en/latest/)

## **Contact Us**

Join the Official Discussion Group on DingTalk

<img src="https://deeprec-dataset.oss-cn-beijing.aliyuncs.com/img/dingtalk_group.JPG" width="200">

Join the Official Discussion Group on WeChat

<img src="https://deeprec-dataset.oss-cn-beijing.aliyuncs.com/img/wechat_group.JPG" width="200">


## **License**

[Apache License 2.0](LICENSE)

