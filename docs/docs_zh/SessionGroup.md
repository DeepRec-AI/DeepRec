# SessionGroup

## 简介
当前Inference场景中，无论用户直接使用TFServing还是使用TF提供的C++接口调用Session::Run，都无法实现多个Session并发处理Request，导致单个Session无法很好的实现CPU or GPU的有效利用。用户如果通过多Instance方式（多进程），无法共享底层的Variable，导致大量使用内存，并且每个Instance各自加载一遍模型，严重影响资源的使用率和模型加载效率。

SessionGroup功能提供了可以配置一组Session，并且将Request通过Round Robin(支持用户自定义策略)方式分发到某一个Session。SessionGroup中的每个Session有私有的线程池，并且支持每个线程池绑定底层的CPU Core，这样可以最大程度的避免共享资源导致的锁冲突开销。SessionGroup中唯一共享的资源是Variable，所有Session共享底层的Variable，并且模型加载只需要加载一次。

通过使用SessionGroup，可以解决内存占用大，但模型CPU使用率低的问题，大大提高资源利用率，在保证latency的前提下极大提高QPS。此外SessionGroup也可以在GPU场景下通过多Session并发执行，大大提高GPU的利用效率。

## 接口介绍

如果用户使用Tensorflow Serving进行服务，可以使用我们提供的代码: [AlibabaPAI/serving](https://github.com/AlibabaPAI/serving/commits/deeprec)，这里已经提供了接入SessionGroup的功能。也可以使用我们提供的[Processor](https://github.com/alibaba/DeepRec/tree/main/serving)代码，Processor没有提供RPC服务框架，需要用户完善，接入自有RPC框架中。

如果用户使用的是自有seving框架，那么需要做的修改如下。

### 1.直接Session::Run进行serving场景
在Inference场景下，如果用户直接使用Session::Run方式实现的Serving，可以参考以下使用方式来使用SessionGroup。

#### 创建SessionGroup

如果是手动创建Session::Run方式实现的Serving，那么将serving框架中NewSession改为NewSessionGroup。
session_num指定SessionGroup中创建多少个Session，用户可以通过评估当前单个Session的CPU利用率，判断需要创建多少个Session。比如如果当前单个Session CPU的最高利用率为20%，建议用户配置4个Session。

```c++

TF_RETURN_IF_ERROR(NewSessionGroup(*session_options_,
    session_group, session_num));
TF_RETURN_IF_ERROR((*session_group)->Create(meta_graph_def_.graph_def()));

```
参考代码: [Processor](https://github.com/alibaba/DeepRec/blob/main/serving/processor/serving/model_session.cc#L143)

#### SessionGroup Run

用户原有代码使用Session::Run可以直接替换为SessionGroup::Run

```c++
status = session_group_->Run(run_options, req.inputs,
    req.output_tensor_names, {}, &resp.outputs, &run_metadata);

```
参考代码: [Processor](https://github.com/alibaba/DeepRec/blob/main/serving/processor/serving/model_session.cc#L308)

用户在processor上使用session_group，只需要在配置文件中增加如下字段：
```c++
"model_config": {
  "session_num": 2,
  "use_per_session_threads": true,
  ...
}
```
更多参数详见：[processor配置参数](https://deeprec.readthedocs.io/zh/latest/Processor.html#id5)

### 2.通过SavedModelBundle进行serving场景
TFServing使用的是SavedModelBundle进行serving的，相关的代码修改参考：[SessionGroup](https://github.com/AlibabaPAI/serving/commit/8b92300da84652f00f13fd20f5df0656cfa26217)，推荐直接使用我们提供的TFServing代码。

#### TFServing中使用SessionGroup

支持SessionGroup的TFServing代码见：[AlibabaPAI/serving](https://github.com/AlibabaPAI/serving/commits/deeprec)

编译文档见：[TFServing编译](https://deeprec.readthedocs.io/zh/latest/TFServing-Compile-And-Install.html)

使用方式如下：
```c++
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --tensorflow_intra_op_parallelism=16 --tensorflow_inter_op_parallelism=16 --use_per_session_threads=true --session_num_per_group=4 --model_base_path=/xxx/pb
```

主要参数：
```c++
session_num_per_group：表示session group中创建几个sessions。

use_per_session_threads：为true表示每个session使用独立的线程池，减少session之间的干扰，建议配置为true。每个session的线程池都是通过tensorflow_intra_op_parallelis和tensorflow_inter_op_parallelism控制大小。
```

用户可以为SessionGroup中每个session指定在哪些cpu cores上执行，默认功能关闭，有两种方式开启：
```
1.用户手动设置，
SESSION_GROUP_CPUSET="2-4;5-7;8-10;11-13"
或者
SESSION_GROUP_CPUSET="2,3,4;5,6,7;8,9,10;11,12,13"
表示有4个session，每个session分别指定cpu上执行。
session0: 2 3 4
session1: 5 6 7
session2: 8 9 10
session3: 11 12 13

2.如果用户不设置环境变量SESSION_GROUP_CPUSET，那么需要设置SET_SESSION_THREAD_POOL_AFFINITY=1，
这样进程会检测哪些cpu可以被分配，从而分给不同的session。
```

## GPU Multi-Stream
在Inference场景中，用户常使用GPU进行线上服务，来提升计算效率，减小延迟。这里可能会遇到的一个问题是，线上GPU利用率低，造成资源浪费。那么为了利用好GPU资源，我们允许用户使用Multi-streams处理请求，在保证延迟的前提下极大提升QPS。

目前multi-streams功能是和SessionGroup功能绑定使用的，SessionGroup的用法详见前面链接。后续我们会在DirectSession上直接支持multi-streams功能。

### 用法
具体用法和SessionGroup的用法一样，在此基础上需要做如下修改。

#### 1.docker启动配置
本优化中使用了GPU MPS(Multi-Process Service)优化，这要求在docker启动后，需要使用下面命令启动后台MPS service进程。

```c++
nvidia-cuda-mps-control -d
```

#### 2.启动命令
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

#### 3.多卡MPS最佳实践
用户机器上可能存在多张GPU卡，对于每个serving instance一般只需要使用一个GPU Device，那么用户可能会在物理机器上启动多个不同的serving instance。在这种情况下使用MPS有一些需要注意的问题，具体如下：

1) 需要在物理机器启动mps daemon进程，这样才有机会让所有的任务docker中对于MPS后台进程可见。

```c++
nvidia-cuda-mps-control -d
```

2) 启动任务docker时，需要增加--ipc=host，保证在docker中对mps daemon进程可见。同时对于每个docker来说，建议mount指定的GPU Device，如下所示：

```c++
sudo docker run -it --name docker_name --ipc=host --net=host --gpus='"device=0"' docker_image bash
```

这样在docker只会可见一张GPU卡，并且逻辑编号为0，那么可以像下面这样执行脚本：

```c++
CUDA_VISIBLE_DEVICES=0 test.py
或者
test.py
```

如果docker mount了所有的GPU Devices，那么在执行脚本的时候，需要手动指定可见的gpu device来达到隔离的效果。

```c++
sudo docker run -it --name docker_name --ipc=host --net=host --gpus=all docker_image bash

docker0中:
CUDA_VISIBLE_DEVICES=0 test.py

docker1中:
CUDA_VISIBLE_DEVICES=1 test.py
```

## 多模型服务
### TF Serving
SessionGroup支持多模型服务，在[TF_serving](https://github.com/AlibabaPAI/serving)上已经支持。对于多模型服务，用户可以对于每个模型服务配置独立的参数，包括不同的模型session group中使用几个session，指定gpu卡，指定线程池等等，从而在框架，资源上进行隔离。

对于GPU任务，目前每个模型只能指定一张GPU卡，所以用户在启动任务时，需要注意GPU资源的划分。

启动多模型服务命令如下：
```c++
CUDA_VISIBLE_DEVICES=0,1 ENABLE_MPS=0 CONTEXTS_COUNT_PER_GPU=4 MERGE_COMPUTE_COPY_STREAM=0 bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --rest_api_port=8888 --use_session_group=true --model_config_file=/data/workspace/serving-model/multi_wdl_model/models.config --platform_config_file=/data/workspace/serving-model/multi_wdl_model/platform_config_file
```
上述命令中最重要的是两个配置文件，

model_config_file:
```
model_config_list:{
    config:{
      name:"pb1",
      base_path:"/data/workspace/serving-model/multi_wdl_model/pb1",
      model_platform:"tensorflow",
      model_id: 0
    },
    config:{
      name:"pb2",
      base_path:"/data/workspace/serving-model/multi_wdl_model/pb2",
      model_platform:"tensorflow",
      model_id: 1
    },
}
```
对于每个模型需要配一个对应的config模块，
* name表示模型服务名称，client端request访问时，需要填充对应的服务名称`request.model_spec.name = 'pb1'`。
* base_path表示模型所在路径。
* model_platform： 默认tensorflow。
* model_id：给每个模型一个编号，从0开始。


platform_config_file:
```
platform_configs {
  key: "tensorflow"
  value {
    source_adapter_config {
      [type.googleapis.com/tensorflow.serving.SavedModelBundleV2SourceAdapterConfig] {
        legacy_config {
          model_session_config {
            session_config {
              gpu_options {
                allow_growth: true
              }
              intra_op_parallelism_threads: 8
              inter_op_parallelism_threads: 8
              use_per_session_threads: true
              use_per_session_stream: true
            }
            session_num: 2
            cpusets: "1,2;5,6"
          }
          model_session_config {
            session_config {
              gpu_options {
                allow_growth: true
              }
              intra_op_parallelism_threads: 16
              inter_op_parallelism_threads: 16
              use_per_session_threads: true
              use_per_session_stream: true
            }
            session_num: 4
            cpusets: "20,21;23,24;26,27;29,30"
          }
        }
      }
    }
  }
}
```
key和上面model_platform字段一样，默认tensorflow。对于每个模型需要配置一个model_session_config，包含session的一些配置。model_session_config最终是一个数组，那么model_session_config[0]即代表model_0的配置，以此类推。


Client端发送请求示例：
```
CUDA_VISIBLE_DEVICES=1,3 ENABLE_MPS=0 MERGE_COMPUTE_COPY_STREAM=0 bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --rest_api_port=8888 --use_session_group=true --model_config_file=/xxx/model_config_file --platform_config_file=/xxx/platform_config_file
```

### EAS+Processor
多模型请求的分发需要EAS框架支持，Processor目前仅支持CPU任务(GPU待支持)，我们可以为不同的模型服务配置不同的cpu资源，线程池等，配置文件如下：
```
{
  "platform": "local",
  "engine": "python",
  "language_type": "PYTHON",
  "name": "pttest",
  "models": [
    {
      "model_path": "https://tf115test.oss-cn-hangzhou.aliyuncs.com/test/libserving_processor_1226_debug3.tar.gz",
      "model_entry": "",
      "name": "model1",
      "processor": "tensorflow",
      "uncompress": true,
      "model_config": {
        "session_num": 2,
        "use_per_session_threads": true,
        "cpusets": "1,2,3;4,5,6",
        "omp_num_threads": 24,
        "kmp_blocktime": 0,
        "feature_store_type": "memory",
        "serialize_protocol": "protobuf",
        "inter_op_parallelism_threads": 24,
        "intra_op_parallelism_threads": 24,
        "init_timeout_minutes": 1,
        "signature_name": "serving_default",
        "model_store_type": "local",
        "checkpoint_dir": "/data/workspace/ckpt/",
        "savedmodel_dir": "/data/workspace/pb/",
        "oss_access_id": "",
        "oss_access_key": "",
        "oss_endpoint": "oss-cn-shanghai.aliyuncs.com"
      }
    },
    {
      "model_path": "https://tf115test.oss-cn-hangzhou.aliyuncs.com/test/libserving_processor_1226_debug3.tar.gz",
      "model_entry": "",
      "name": "model2",
      "processor": "tensorflow",
      "uncompress": true,
      "model_config": {
        "session_num": 4,
        "use_per_session_threads": true,
        "cpusets": "7-9;10-12;13-15;16-18",
        "omp_num_threads": 24,
        "kmp_blocktime": 0,
        "feature_store_type": "memory",
        "serialize_protocol": "protobuf",
        "inter_op_parallelism_threads": 24,
        "intra_op_parallelism_threads": 24,
        "init_timeout_minutes": 1,
        "signature_name": "serving_default",
        "model_store_type": "local",
        "checkpoint_dir": "/data/workspace/ckpt2/",
        "savedmodel_dir": "/data/workspace/pb2/",
        "oss_access_id": "",
        "oss_access_key": "",
        "oss_endpoint": "oss-cn-shanghai.aliyuncs.com"
      }
    }
  ],
  "processors": [
    {
      "name": "tensorflow",
      "processor_path": "https://tf115test.oss-cn-hangzhou.aliyuncs.com/test/libserving_processor_1226_debug3.tar.gz",
      "processor_entry": "libserving_processor.so",
      "processor_type": "cpp"
    }
  ],
  "metadata":{
    "cpu":32,
    "eas":{
      "enabled_model_verification":false,
      "scheduler":{
        "enable_cpuset": false
      }
    },
    "gpu":0,
    "instance":1,
    "memory":40960,
    "rpc":{
      "io_threads":10,
      "worker_threads":20,
      "enable_jemalloc":true
    }
  },
}
```
上面配置文件和单模型下的区别在于多了“models”和“processors”字段。

“processors”字段是一个list，用户可以配置多个processor，在models中对于不同的模型可以配置不同的processor。

“models”字段是一个list，用户可以配置多个模型服务，每个模型服务都有单独的配置，主要是“model_config”字段，更详细的字段介绍见：[model config](https://deeprec.readthedocs.io/zh/latest/Processor.html#id5)

用户Client请求示例：
```
client = PredictClient('127.0.0.1:8080', 'pttest/model1')
#client = PredictClient('127.0.0.1:8080', 'pttest/model2')
client.init()

pb_string = open("./warm_up.bin", "rb").read()
request = TFRequest()
request.request_data.ParseFromString(pb_string)
request.add_fetch("dinfm/din_out/Sigmoid:0")

resp = client.predict(request)
```
在构建PredictClient时需要在url增加模型名称，如"pttest/model1"。
