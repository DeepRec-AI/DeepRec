# DeepRec源代码编译&安装

## 开发环境准备

**CPU Base Docker Image**

```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-base-cpu-py36-ubuntu18.04
```

Docker Hub repository
```
alideeprec/deeprec-base:deeprec-base-cpu-py36-ubuntu18.04
```

**GPU(cuda11.0) Base Docker Image**

```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-base-gpu-py36-cu110-ubuntu18.04
```

Docker Hub repository
```
alideeprec/deeprec-base:deeprec-base-gpu-py36-cu110-ubuntu18.04
```

**CPU Dev Docker (with bazel cache)**

```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-cpu-py36-ubuntu18.04
```

Docker Hub repository
```
alideeprec/deeprec-build:deeprec-dev-cpu-py36-ubuntu18.04
```

**GPU(cuda11.0) Dev Docker (with bazel cache)**

```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-developer:deeprec-dev-gpu-py36-cu110-ubuntu18.04
```

Docker Hub repository
```
alideeprec/deeprec-build:deeprec-dev-gpu-py36-cu110-ubuntu18.04
```

## 代码编译

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

**GPU CUDA11.0镜像**

```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-training:deeprec2204u1-gpu-py36-cu110-ubuntu18.04
```

Docker Hub repository
```
alideeprec/deeprec-release:deeprec2204u1-gpu-py36-cu110-ubuntu18.04
```

**CPU镜像**

```
registry.cn-shanghai.aliyuncs.com/pai-dlc-share/deeprec-training:deeprec2204u1-cpu-py36-ubuntu18.04
```

Docker Hub repository
```
alideeprec/deeprec-release:deeprec2204u1-cpu-py36-ubuntu18.04
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
