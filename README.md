
![DeepRec Logo](https://github.com/alibaba/DeepRec/blob/main/docs/deeprec_logo.png)

--------------------------------------------------------------------------------

## **Introduction**
DeepRec is a recommendation engine based on [TensorFlow 1.15](https://www.tensorflow.org/), [Intel-TensorFlow](https://github.com/Intel-tensorflow/tensorflow) and [NVIDIA-TensorFlow](https://github.com/NVIDIA/tensorflow).


### **Background**
Sparse model is a type of deep learning model that accounts for a relatively high proportion of discrete feature calculation logic in the model structure. Discrete features are usually expressed as non-numeric features that cannot be directly processed by algorithms such as id, tag, text, and phrases. They are widely used in high-value businesses such as search, advertising, and recommendation.


DeepRec has been deeply cultivated since 2016, which supports core businesses such as Taobao Search, recommendation and advertising. It precipitates a list of features on basic frameworks and has excellent performance in sparse models training. Facing a wide variety of external needs and the environment of deep learning framework embracing open source, DeepeRec open source is conducive to establishing standardized interfaces, cultivating user habits, greatly reducing the cost of external customers working on cloud and establishing the brand value.

### **Key Features**
DeepRec has super large-scale distributed training capability, supporting model training of trillion samples and 100 billion Embedding Processing. For sparse model scenarios, in-depth performance optimization has been conducted across CPU and GPU platform. It contains 3 kinds of features to improve usability and performance for super-scale scenarios. 
#### **Sparse Functions**
 - Embedding Variable.
 - Dynamic Dimension Embedding Variable.
 - Adaptive Embedding Variable.
 - Multiple Hash Embedding Variable.
 #### **Performance Optimization**
 - Distributed Training Framework Optimization, such as grpc+seastar, FuseRecv, StarServer, HybridBackend etc.
 - Runtime Optimization, such as CPU memory allocator (PRMalloc), GPU memory allocator etc.
 - Operator level optimization, such as BF16 mixed precision  optimization, sparse operator optimization and EmbeddingVariable on PMEM and GPU, new hardware feature enabling, etc.
 - Graph level optimization, such as AutoGraphFusion, SmartStage, AutoPipeline, StrutureFeature, MicroBatch etc.
#### **Deploy and Serving**
 - Incremental model loading and exporting
 - Super-scale sparse model distributed serving
 - Multilevel hybrid storage and multi backend supported ..
 - Online deep learning with low latency


***
## **Installation**


### **Prepare for installation**


CPU Platform

```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-cpu-py36-ubuntu18.04
```

GPU Platform


```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-gpu-py36-cu110-ubuntu18.04
```
### **How to Build**

configure
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
$ bazel build  -c opt --config=opt  --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true //tensorflow/tools/pip_package:build_pip_package
```
Compile for CPU optimization and ABI=0
```
$ bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --config=opt --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true //tensorflow/tools/pip_package:build_pip_package
```
### **Create whl package** 
```
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```
### **Install whl package**
```
$ pip3 install /tmp/tensorflow_pkg/tensorflow-1.15.5+${version}-cp36-cp36m-linux_x86_64.whl
```

### **How to Build serving library**

configure will modify .bazelrc file, please revert the change when you build DeepRec whl.
```
./configure serving
```
Or configure with some flags,
```
./configure serving --mkl
./configure serving --mkl_open_source_v1_only
./configure serving --mkl_threadpool
./configure serving --mkl --cuda ...
```
More details see: serving/configure.py

build processor library, this will generate libserving_processor.so. User should load the library, then call serving API to predict.
```
bazel build //serving/processor/serving:libserving_processor.so
```

UT test
```
bazel test -- //serving/processor/... -//serving/processor/framework:lookup_manual_test
```

End2End test
```
Details please see: serving/processor/tests/end2end/README
```

### **Nightly Images**
#### Image for GPU CUDA11.0
```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-training:deeprec-nightly-gpu-py36-cu110-ubuntu18.04
```
#### Image for CPU
```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-training:deeprec-nightly-cpu-py36-ubuntu18.04
```


***
## Continuous Build Status

### Official Build

| Build Type    | Status                                                       |
| ------------- | ------------------------------------------------------------ |
| **Linux CPU** | ![CPU Build](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-build-wheel.yaml/badge.svg) |
| **Linux GPU** | ![GPU Build](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cuda11.2-cibuild-build-wheel.yaml/badge.svg) |
| **Linux CPU Serving** | ![CPU Serving Build](https://github.com/alibaba/DeepRec/actions/workflows/ubuntu18.04-py3.6-cibuild-build-serving.yaml/badge.svg) |

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

## **User Document (Chinese)**

[https://deeprec.rtfd.io](https://deeprec.rtfd.io)

## **Contact Us**

Join the Official Discussion Group on DingTalk

<img src="docs/README/deeprec_dingtalk.png" width="200">

## **License**

[Apache License 2.0](LICENSE)

