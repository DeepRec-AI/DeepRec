# TFServing源代码编译&安装

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

## TFServing代码库及分支

我们提供了针对DeepRec版本的TFServing，该版本指向DeepRec Repo.

代码库：[https://github.com/AlibabaPAI/serving](https://github.com/AlibabaPAI/serving)
开发分支：deeprec
Latest Release Tag：r1.15.5-deeprec2204u1

## TFServing编译&打包

**代码编译-CPU版本**

```bash
bazel build -c opt tensorflow_serving/...
```

**代码编译-GPU版本**

```bash
bazel build -c opt --config=cuda tensorflow_serving/...
```

**生成Client Wheel包**

```bash
bazel-bin/tensorflow_serving/tools/pip_package/build_pip_package /tmp/tf_serving_client_whl
```

**Server Bin**

Server Bin生成在下面路径中：
```bash
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```
