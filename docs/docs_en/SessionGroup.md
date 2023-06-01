# SessionGroup

## Introduction
In the current recommendation inference scenario, the user always used the tensorflow_serving framework or EAS+Processor which is a serving framework in Alibaba. There is usually only one session in the process of these frameworks, and a single session makes it impossible to efficiently utilize resources such as CPU and GPU. Of course, users can perform tasks in a multi-instance mode (multi-process), but this method cannot share the Variables, resulting in a large amount of memory usage, and each Instance loads the model once, seriously affecting resource usage and model loading efficiency.

By using SessionGroup, it can solve the problem of large memory usage but low model CPU usage, greatly improve resource utilization, and greatly improve QPS under the premise of ensuring latency. In addition, multiple sessions in SessionGroup can also be executed concurrently in GPU scenarios, which greatly improves the utilization efficiency of GPUs.

## Usage
If users use tensorflow_serving for services, they can use the code we provide: [DeepRec-AI/serving](https://github.com/DeepRec-AI/serving/commits/deeprec), here has already provided the function of accessing SessionGroup. You can also use the [Processor](https://github.com/alibaba/DeepRec/tree/main/serving) code provided by us. Processor does not provide an RPC service framework, you can use our RPC framework [PAI-EAS](https://www.aliyun.com/activity/bigdata/pai/eas) or yours.

### 1.Processor + EAS
#### CPU Task
If users use session_group on EAS processor, they only need to add the following fields in the configuration file:
```c++
"model_config": {
  "session_num": 2,
  "use_per_session_threads": true,
  ...
}
```

#### GPU Task
For GPU tasks, the following configurations are required:
```c++
"model_config": {
  "session_num": 2,
  "use_per_session_threads": true,
  "gpu_ids_list": "0,2",
  ...
}
```

More parameters see: [processor configuration parameters](https://deeprec.readthedocs.io/zh/latest/Processor.html#id5)

### 2.Tensorflow serving
Our offered tensorflow_serving now do serving via SavedModelBundle. We already support SessionGroup in tensorflow_serving, related code modification reference: [SessionGroup](https://github.com/DeepRec-AI/serving/commit/8b92300da84652f00f13fd20f5df0656cfa26217), it is recommended to use the tensorflow_serving code provided by us directly.

tensorflow_serving repo that supports SessionGroup: [DeepRec-AI/serving](https://github.com/DeepRec-AI/serving/commits/deeprec)

Compilation documentation see: [TFServing compilation](https://deeprec.readthedocs.io/zh/latest/TFServing-Compile-And-Install.html)

#### CPU Task
Run command:
```c++
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --tensorflow_intra_op_parallelism=16 --tensorflow_inter_op_parallelism=16 --use_per_session_threads=true --session_num_per_group=4 --model_base_path=/xxx/pb
```

Releated Args:
```c++
session_num_per_group: Indicates how many sessions are created in the SessionGroup.

use_per_session_threads: If it is true, it means that each session uses an independent thread pool to reduce the interference between sessions. It is recommended to configure it as true. The thread pool size of each session is controlled by tensorflow_intra_op_parallelis and tensorflow_inter_op_parallelism.
```

Users can specify the CPU cores for each session in the SessionGroup. The default function is disabled. There are two ways to enable it:
```
1.User manually sets environment variables:
SESSION_GROUP_CPUSET="2-4;5-7;8-10;11-13"
Or
SESSION_GROUP_CPUSET="2,3,4;5,6,7;8,9,10;11,12,13"
This indicates that there are 4 sessions, and each session is executed on the specified CPU.
session0: 2 3 4
session1: 5 6 7
session2: 8 9 10
session3: 11 12 13

2.User needs to set SET_SESSION_THREAD_POOL_AFFINITY=1 if they does not set the environment variable SESSION_GROUP_CPUSET.
DeepRec will detect which CPUs can be allocated, and then allocate CPU cores to different sessions according to the distribution of CPUs on NUMA nodes.
```

These options can be used in GPU task.

#### GPU Task
In Inference scenarios, users often use GPUs for online services to improve computing efficiency and reduce latency. One problem that may be encountered here is that the online GPU utilization rate is low, resulting in a waste of resources. Then, to make good use of GPU resources, we use Multi-streams to process requests, which greatly improves QPS while ensuring latency. In the GPU scenario, using session group will use multi-stream by default, that is, each session uses an independent stream.

At present inference scenarios, the multi-streams function is bound to the SessionGroup function. For the usage of SessionGroup, see the previous link. In the future, we will directly support the multi-streams function on DirectSession.

The specific usage is the same as that of SessionGroup, and the following modifications need to be made on this basis.

##### 1.Docker startup configuration
This function uses GPU MPS (Multi-Process Service) optimization(optional, recommended to enable), which requires the background MPS service process to be started in the docker container.
```c++
nvidia-cuda-mps-control -d
```

##### 2.Startup command
Currently taking Tensorflow_serving as an example (it will be necessary to add other framework usage methods later), the following parameters need to be added when starting the server.

```c++
CUDA_VISIBLE_DEVICES=0  ENABLE_MPS=1 CONTEXTS_COUNT_PER_GPU=4 MERGE_COMPUTE_COPY_STREAM=1 PER_SESSION_HOSTALLOC=1 bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --tensorflow_intra_op_parallelism=8 --tensorflow_inter_op_parallelism=8 --use_per_session_threads=true  --session_num_per_group=4 --allow_gpu_mem_growth=true --model_base_path=/xx/xx/pb/

ENABLE_MPS=1: Turn on MPS (it is generally recommended to turn on).
CONTEXTS_COUNT_PER_GPU=4: Configure cuda contexts count for each physical GPU, the default is 4.
MERGE_COMPUTE_COPY_STREAM: The calculation operator and the copy operator use the
                           same stream to reduce waiting between different streams.
PER_SESSION_HOSTALLOC=1: Each session uses an independent gpu host allocator.

use_per_session_threads=true: Each session configures the thread pool separately.
session_num_per_group=4: Indicates the number of sessions configured by the session group.
```

##### 3.Multi-GPU
If the user does not specify CUDA_VISIBLE_DEVICES=0 and there are multiple GPUs on the machine, the session group will use all GPUs by default. Assuming there are 2 GPUs, and session_num_per_group=4 is set, then the session group will create 4 streams on each GPU, because currently a stream corresponds to a session, so there are a total of 2*4=8 sessions in the current session group. The model parameters required by these sessions on the CPU are all shared. For the model parameters of the place on the GPU, if the stream associated with the session is on the same GPU, then the GPU parameters are shared between these sessions, otherwise the sessions don't share GPU parameters.

Users can specify which physical GPUs are assigned to a session group,
```
--gpu_ids_list=0,2
```
The option above indicate that GPUs 0 and 2 are assigned to the current session group. It should be noted that the GPU number here does not correspond to the number seen by nvidia-smi, but the number seen by deeprec.

For example, suppose there are 4 GPUs on the physical machine, and the numbers are 0, 1, 2, 3. If the user set nothing, then the numbers 0, 1, 2, 3 and the physical GPU numbers seen in deeprec are the same. If the user sets CUDA_VISIBLE_DEVICES=3,2,1,0, then the physical GPU numbers corresponding to the numbers 0,1,2,3 seen in deeprec are 3,2,1,0 respectively.

The corresponding relationship above does not affect the actual use. Users only need to care about how many physical GPUs deeprec can actually see. Assuming that 3 GPUs can be seen, then the visible numbers of deeprec are 0, 1, and 2.

The option --gpu_ids_list=0,2 means that the user can use GPUs 0 and 2. If the number of GPUs visible to the process is less than 3 (0,1,2), an error will be reported.

Startup command:
```
ENABLE_MPS=1 CONTEXTS_COUNT_PER_GPU=4 bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --tensorflow_intra_op_parallelism=8 --tensorflow_inter_op_parallelism=8 --use_per_session_threads=true  --session_num_per_group=4 --allow_gpu_mem_growth=true --gpu_ids_list=0,2  --model_base_path=/xx/xx/pb/
```
For a detailed explanation of the above environment variables, see: [Startup parameters](https://deeprec.readthedocs.io/en/latest/SessionGroup.html#startup-command)

##### 4.Best practice for using MPS with multiple GPU
There may be multiple GPUs on the user machine, and generally only one GPU Device is required for each serving instance, so the user may start multiple different serving instances on the physical machine. There are some issues to be aware of when using MPS in this situation, as follows:

1) The MPS daemon process needs to be started on the physical machine so that all tasks in docker can be managed by the MPS background process.

```c++
nvidia-cuda-mps-control -d
```

2) When starting docker, you need to add '--ipc=host' to ensure that the process in docker is visible to the MPS daemon process. At the same time, for each docker, it is recommended to mount the specified GPU Device, as follows:

```c++
sudo docker run -it --name docker_name --ipc=host --net=host --gpus='"device=0"' docker_image bash
```

In this way, only one GPU will be visible in docker, and the logical number is 0, then the script can be executed as follows:

```c++
CUDA_VISIBLE_DEVICES=0 test.py
Or
test.py
```

If docker mounts all GPU Devices, then when executing the script, you need to manually specify the visible GPU device to achieve the effect of resource isolation.

```c++
sudo docker run -it --name docker_name --ipc=host --net=host --gpus=all docker_image bash

docker0 task:
CUDA_VISIBLE_DEVICES=0 test.py

docker1 task:
CUDA_VISIBLE_DEVICES=1 test.py
```

The tensorflow_serving code modified by DeepRec is as follows:: [TF serving](https://github.com/DeepRec-AI/serving/commits/deeprec)

### 3.Use user-defined framework
If users want to implement the session group function into their own framework, they can refer to the code implementation in the processor below.

#### Create SessionGroup
If you manually create a serving framework implemented by Session::Run, then modify NewSession API in the serving framework to NewSessionGroup. 
session_num specifies how many sessions are created in the SessionGroup, user can judge how many sessions need to be created by evaluating the CPU utilization of the current single session. For example, if the current maximum CPU utilization of a single session is 20%, it is recommended that users configure 5 sessions.

```c++
TF_RETURN_IF_ERROR(NewSessionGroup(*session_options_,
    session_group, session_num));
TF_RETURN_IF_ERROR((*session_group)->Create(meta_graph_def_.graph_def()));
```
Reference Code: [Processor](https://github.com/alibaba/DeepRec/blob/main/serving/processor/serving/model_session.cc#L143)

#### SessionGroup Run API
The Session::Run API used by users can be directly replaced by SessionGroup::Run.
```c++
status = session_group_->Run(run_options, req.inputs,
    req.output_tensor_names, {}, &resp.outputs, &run_metadata);
```
Reference Code: [Processor](https://github.com/alibaba/DeepRec/blob/main/serving/processor/serving/model_session.cc#L308)

## Multi-model service
### TF Serving
SessionGroup supports multi-model services, which have been supported on [TF_serving](https://github.com/DeepRec-AI/serving). For multi-model services, users can configure independent parameters for each model service, including the number of sessions in session groups in different model services, specifying GPUs, thread pools, etc., so as to isolate frameworks and resources.

For GPU tasks, the session group can specify one or more GPUs, so users need to pay attention to the division of GPU resources when starting multi-model tasks.

The command to start the multi-model service is as follows:
```c++
ENABLE_MPS=1 CONTEXTS_COUNT_PER_GPU=4 MERGE_COMPUTE_COPY_STREAM=1 PER_SESSION_HOSTALLOC=1 bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --rest_api_port=8888 --use_session_group=true --model_config_file=/data/workspace/serving-model/multi_wdl_model/models.config --platform_config_file=/data/workspace/serving-model/multi_wdl_model/platform_config_file
```
For a detailed explanation of the above environment variables, see: [Startup parameters](https://deeprec.readthedocs.io/en/latest/SessionGroup.html#startup-command)

The following are the two most important configuration files in the command,

Assuming that there are 4 GPU devices on the machine, the configuration is as follows:

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
For each model service, a corresponding model_config_file is required
* name: indicates the name of the model service. When requesting access from the client side, the corresponding service name `request.model_spec.name = 'pb1'` needs to be filled.
* base_path: indicates the path where the model is located.
* model_platform: default value 'tensorflow'.
* model_id: Give each model service a number, starting from 0.


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
            gpu_ids: [0,1]
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
            session_num: 2
            cpusets: "20,21;23,24;26,27;29,30"
            gpu_ids: [2,3]
          }
        }
      }
    }
  }
}
```
The key is the same as the model_platform field above, and the default value is ‘tensorFlow’. For each model service, a model_session_config needs to be configured, including some configurations of the session. model_session_config is finally an array, then model_session_config[0] represents the configuration of model_0 service, and so on.

Note the gpu_ids parameter above, indicating which GPUs are used by each session group. Assume here that there are 4 GPUs on the machine, then use gpu-0, gpu1 and gpu-2, gpu-3 on the two session groups respectively. Special attention needs to be paid. At this time, the total number of sessions in the session group is session_num*gpu_ids.size, that is, 4 sessions.

Server example:
```
CUDA_VISIBLE_DEVICES=1,3 ENABLE_MPS=1 MERGE_COMPUTE_COPY_STREAM=1 PER_SESSION_HOSTALLOC=1 bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --rest_api_port=8888 --use_session_group=true --model_config_file=/xxx/model_config_file --platform_config_file=/xxx/platform_config_file
```
For a detailed explanation of the above environment variables, see: [Startup parameters](https://deeprec.readthedocs.io/en/latest/SessionGroup.html#startup-command)

Client example:
```
...
request = predict_pb2.PredictRequest()
request.model_spec.name = 'pb2' # set model name here, like 'pb1', 'pb2' ...
request.model_spec.signature_name = 'serving_default'
...
```

### EAS+Processor
The distribution of multi-model requests requires the support of the EAS framework. We can configure different cpu resources, thread pools, etc. for different model services. The configuration files are as follows:
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
The difference between the multi-model service configuration file and the single-model configuration file is that the "models" and "processors" fields are added.

The "processors" field is a list. Users can configure multiple processors. In the "models" field, different processors can be configured for different model services.

The "models" field is a list. Users can configure fields separately for multiple model services. Each model service has a separate configuration, mainly the "model_config" field. For more detailed field introductions, see:

Example:
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
When building 'PredictClient', you need to add the model name to the url, such as "pttest/model1".

