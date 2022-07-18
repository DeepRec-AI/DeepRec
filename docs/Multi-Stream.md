# Multi-Stream

## 简介
在Inference场景中，用户常使用GPU进行线上服务，来提升计算效率，减小延迟。这里可能会遇到的一个问题是，线上GPU利用率低，造成资源浪费。那么为了利用好GPU资源，我们允许用户使用Multi-streams处理请求，在保证延迟的前提下极大提升QPS。

目前我们的multi-streams功能是和[SessionGroup](https://deeprec.readthedocs.io/zh/latest/SessionGroup.html)功能绑定使用的，SessionGroup的用法详见前面链接。后续我们会在DirectSession上直接支持multi-streams功能。

## 接口介绍

具体用法和SessionGroup的用法一样，在此基础上需要做如下修改。

### 1.docker启动配置
本优化中使用了GPU MPS(Multi-Process Service)优化，这要求在docker启动后，需要使用下面命令启动后台MPS service进程。

```c++
nvidia-cuda-mps-control -d
```

### 2.启动命令
这里以Tensorflow serving为例(后续补充其他使用方式)，在启动server时需要增加下列参数，

```c++
CUDA_VISIBLE_DEVICES=0  ENABLE_MPS=1 CONTEXTS_COUNT_PER_GPU=4 bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --tensorflow_intra_op_parallelism=8 --tensorflow_inter_op_parallelism=8 --use_per_session_threads=true  --session_num_per_group=4 --use_multi_stream=true --allow_gpu_mem_growth=true --model_base_path=/xx/xx/pb/

ENABLE_MPS=1: 开启MPS(一般都建议开启)。
CONTEXTS_COUNT_PER_GPU=4: 每个物理GPU配置几组cuda context，默认是4。
use_per_session_threads=true: 每个session单独配置线程池。
session_num_per_group=4: session group中配置几个session。
use_multi_stream=true: 开启multi-stream功能。
```
TF serving用DeepRec提供的代码: [TF serving](https://github.com/AlibabaPAI/serving/commits/deeprec)

