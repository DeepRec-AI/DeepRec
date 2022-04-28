# Estimator源代码编译&安装

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

## Estimator代码库及分支

由于DeepRec新增了分布式grpc++、star_server等protocol，在使用DeepRec配合原生Estimator会存在像grpc++, star_server功能使用时无法通过Estimator检查的问题，因为我们提供了针对DeepRec版本的Estimator.

代码库：[https://github.com/AlibabaPAI/estimator](https://github.com/AlibabaPAI/estimator)
分支：deeprec

## Estimator编译

**代码编译**

```bash
bazel build //tensorflow_estimator/tools/pip_package:build_pip_package
```

**生成Wheel包**

```bash
bazel-bin/tensorflow_estimator/tools/pip_package/build_pip_package /tmp/estimator_whl
```

## Estimator安装

安装DeepRec会默认安装原生的tensorflow-estimator的版本，请重装新编译的tensorflow-estimator即可。

