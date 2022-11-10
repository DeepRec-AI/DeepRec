# DeepRec源代码编译&安装

## 开发环境准备

**CPU Base Docker Image**

| GCC Version | Python Version |                           IMAGE                           |
| ----------- | -------------- | --------------------------------------------------------- |
|   7.5.0     |    3.6.9       | alideeprec/deeprec-base:deeprec-base-cpu-py36-ubuntu18.04 |
|   11.2.0    |    3.8.6       | alideeprec/deeprec-base:deeprec-base-cpu-py38-ubuntu22.04 |


**GPU Base Docker Image**

| GCC Version | Python Version | CUDA VERSION |                           IMAGE                                 |
| ----------- | -------------- | ------------ | --------------------------------------------------------------- |
|    7.5.0    |    3.6.9       | CUDA 11.0.3  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu110-ubuntu18.04 |
|    7.5.0    |    3.6.9       | CUDA 11.2.2  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu112-ubuntu18.04 |
|    7.5.0    |    3.6.9       | CUDA 11.4.2  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu114-ubuntu18.04 |
|    7.5.0    |    3.6.9       | CUDA 11.6.1  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu116-ubuntu18.04 |
|    7.5.0    |    3.6.9       | CUDA 11.7.1  | alideeprec/deeprec-base:deeprec-base-gpu-py36-cu117-ubuntu18.04 |
|    11.2.0   |    3.8.6       | CUDA 11.7.1  | alideeprec/deeprec-base:deeprec-base-gpu-py38-cu117-ubuntu22.04 |

**CPU Dev Docker (with bazel cache)**

```
alideeprec/deeprec-build:deeprec-dev-cpu-py36-ubuntu18.04
```

**GPU(cuda11.6) Dev Docker (with bazel cache)**

```
alideeprec/deeprec-build:deeprec-dev-gpu-py36-cu116-ubuntu18.04
```

## 代码编译

**GPU Environment**
为了更好的发挥GPU性能，根据编译/运行的GPU卡，配置不同的TF_CUDA_COMPUTE_CAPABILITIES

| GPU architecture    | TF_CUDA_COMPUTE_CAPABILITIES |
| ------------------- | ---------------------------- |
| Pascal (P100)       | 6.0+6.1                      |
| Volta (V100)        | 7.0                          |
| Turing (T4)         | 7.5                          |
| Ampere (A10, A100)  | 8.0+8.6                      |

如果希望编译出支持不同GPU卡上执行的版本，可以配置多个值，比如DeepRec中默认配置为"6.0,6.1,7.0,7.5,8.0"

比如配置环境变量TF_CUDA_COMPUTE_CAPABILITIES方法：

```bash
export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0"
```

```bash
./configure
```

**GPU/CPU版本编译**

```bash
bazel build -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
```

**GPU/CPU版本编译+ABI=0**

```bash
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --config=opt //tensorflow/tools/pip_package:build_pip_package
```

**编译开启OneDNN + Eigen Threadpool工作线程池版本（CPU）**

```bash
bazel build  -c opt --config=opt  --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true //tensorflow/tools/pip_package:build_pip_package
```

**编译开启OneDNN + Eigen Threadpool工作线程池版本+ABI=0的版本 （CPU）**

```bash
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" -c opt --config=opt --config=mkl_threadpool --define build_with_mkl_dnn_v1_only=true //tensorflow/tools/pip_package:build_pip_package
```

## 生成Whl包

```bash
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

## 安装Whl包

```bash
pip3 install /tmp/tensorflow_pkg/tensorflow-1.15.5+${version}-cp36-cp36m-linux_x86_64.whl
```

## 最新Release镜像

**CPU镜像**

```
alideeprec/deeprec-release:deeprec2208u1-cpu-py36-ubuntu18.04
```

**GPU CUDA11.6镜像**

```
alideeprec/deeprec-release:deeprec2208u1-gpu-py36-cu116-ubuntu18.04
```

## DeepRec Processor编译打包

配置.bazelrc（注意如果编译DeepRec请重新进行configure配置）
```
./configure serving
```
增加MKL相关配置
```
./configure serving --mkl
./configure serving --mkl_open_source_v1_only
./configure serving --mkl_threadpool
./configure serving --mkl --cuda ...
```
更多细节请查看: serving/configure.py

编译processor库，会生成libserving_processor.so，用户可以加载该so，并且调用示例中的serving API进行predict
```
bazel build //serving/processor/serving:libserving_processor.so
```
单元测试
```
bazel test -- //serving/processor/... -//serving/processor/framework:lookup_manual_test
```
E2E测试细节请查看: serving/processor/tests/end2end/README
