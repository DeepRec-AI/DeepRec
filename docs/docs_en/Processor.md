# Processor

## Introduction
DeepRec Serving Processor is a library for online high-performance services. It is based on DeepRec and support the following functions:

● Supports automatic discovery and import of full models;

● Support incremental model updates, reduce the time consumption caused by loading the model, and make the model online more real-time;

● Supports asynchronous update of models to reduce online serving performance jitter;

● Support model rollback to different versions;

● Support model WarmUp, avoiding the problem of a slow start when loading the model for the first time;

● Support model local storage (memory + Pmem + SSD) and distributed storage services (multi-node shared storage model parameters), etc.

● Compatible with native Tensorflow, it can also serve normally for graphs trained with native Tensorflow.

● The easy-to-use API interface makes it easier for users to use.

The Processor is an independent .so file that users can easily link to their own Serving RPC framework.

## Compile
Compile details[compile processor](https://deeprec.readthedocs.io/zh/latest/DeepRec-Compile-And-Install.html#deeprec-processor), we will get “**libserving_processor.so**” after compiling.

## Usage
Users can use it in two ways:

First, directly [dlopen](https://linux.die.net/man/3/dlopen) in the user framework code to load the symbols in so.

Second, use the header files "**serving/processor/serving/processor.h**" and "**libserving_processor.so**".

**Attention**: If you are not using DeepRec docker, then some additional .so dependencies may be required, including: libiomp5.so，libmklml_intel.so，libstdc++.so.6.

#### C API
Processor provides the following C API interfaces, and users need to call the following interfaces in their Serving framework.

**1) initialize**
```c
void* initialize(const char* model_entry, const char* model_config, int* state);
```
**Args:**

model_entry: By default, the string "" is passed (note that it is not NULL).

model_config: The json content read from the configuration file.

state: Return to the user the status of the Serving framework, 0 is normal, -1 is abnormal.

**Return value:**
Return a pointer, which is the model_buf argument in the process function below. This argument needs to be passed in each time the process funtion is called.

**Usage:**
The initialize function is called once when the Serving RPC framework starts. The framework needs to save the returned pointer, and this parameter needs to be passed in when the process function is subsequently called.

**Example:**
```c
const char* entry = "xxx";
const char* config = "json configs";
int state = -1;
void* model = initialize(entry, config, &state);
```

**2) process**
```c
int process(void* model_buf, const void* input_data, int input_size, void** output_data, int* output_size);
```
**Args:**

model_buf: The returned pointer value of initialize function.

input_data: The user request serialized into a byte stream by protobuf, the protobuf format is shown in PredictRequest "data format" below.

input_size: The size of the request.

output_data: The result returned by the prediction is a serialized byte stream in the protobuf format. For the protobuf format, see PredictResponse "data format" below. (Note: The returned buffer is allocated on the heap memory, and the user framework needs to be responsible for reclaiming the memory, otherwise there will be a memory leak.)

output_size: The size of the response.

**Return value:**
Return status code, 200 means OK, 500 means service error.

**Usage:**
The user Serving framework receives the request sent by the Client. If the requested data format is valid for the Processor (see "Data Format" below), it can directly call the Process function to perform prediction. If the data format is invalid, it needs to be converted into the format required by the Processor (see "Data Format" below) and then call the Process function.

**Example:**
```c
void* model = initialize(xx, xx, xx);
...
char* input_data = "xxx";
int input_size = 3;
void* output_data = nullptr;
int output_size = 0;
int state = process(model, (void*)input_data, input_size, &output_data, &output_size);
```

**3) get_serving_model_info**
```c
int get_serving_model_info(void* model_buf, void** output_data, int* output_size);
```
**Args:**

model_buf: The returned pointer value of initialize function.

output_data: The returned result is a serialized byte stream in protobuf format. For the protobuf format, see ServingModelInfo "data format" below. (Note: The returned buffer is allocated on the heap memory, and the user framework needs to be responsible for reclaiming the memory, otherwise there will be a memory leak.)

output_size: The size of output_data.

**Return value**
Return status code, 200 means OK, 500 means service error.

**Usage:**
Users can call this API query when they need information about the model currently being served.

**Example:**
```c
void* model = initialize(xx, xx, xx);
...
void* output_data = nullptr;
int output_size = 0;
int state = get_serving_model_info(model, &output_data, &output_size);
```

#### data format
The Request, Response and other data formats required by the Processor are as follows. Here we use Protobuf as the data storage format. Consistent with "**serving/processor/serving/predict.proto**" under DeepRec.
Protobuf is the abbreviation of Protocol Buffers. It is a data description language used to describe a portable and efficient structured data storage format. Protobuf can be used for structured data serialization or serialization. Simply put, data can be parsed from one language to another. After Java's Protobuf data is serialized, it can be parsed in C++ as it is, which facilitates data exchange in various scenarios.
The user needs to encapsulate the data into the following format on the client side, or convert it to this format before call Process function.
```c
enum ArrayDataType {
  // Not a legal value for DataType. Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0;
  
  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1;
  DT_DOUBLE = 2;
  DT_INT32 = 3;
  DT_UINT8 = 4;
  DT_INT16 = 5;
  DT_INT8 = 6;
  DT_STRING = 7;
  DT_COMPLEX64 = 8;  // Single-precision complex
  DT_INT64 = 9;
  DT_BOOL = 10;
  DT_QINT8 = 11;     // Quantized int8
  DT_QUINT8 = 12;    // Quantized uint8
  DT_QINT32 = 13;    // Quantized int32
  DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15;    // Quantized int16
  DT_QUINT16 = 16;   // Quantized uint16
  DT_UINT16 = 17;
  DT_COMPLEX128 = 18;  // Double-precision complex
  DT_HALF = 19;
  DT_RESOURCE = 20;
  DT_VARIANT = 21;  // Arbitrary C++ data types
}

// Dimensions of an array
message ArrayShape {
  repeated int64 dim = 1 [packed = true];
}

// Protocol buffer representing an array
message ArrayProto {
  // Data Type.
  ArrayDataType dtype = 1;

  // Shape of the array.
  ArrayShape array_shape = 2;

  // DT_FLOAT.
  repeated float float_val = 3 [packed = true];

  // DT_DOUBLE.
  repeated double double_val = 4 [packed = true];

  // DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
  repeated int32 int_val = 5 [packed = true];

  // DT_STRING.
  repeated bytes string_val = 6;

  // DT_INT64.
  repeated int64 int64_val = 7 [packed = true];

  // DT_BOOL.
  repeated bool bool_val = 8 [packed = true];
}

// PredictRequest specifies which TensorFlow model to run, as well as
// how inputs are mapped to tensors and how outputs are filtered before
// returning to user.
message PredictRequest {
  // A named signature to evaluate. If unspecified, the default signature
  // will be used
  string signature_name = 1;

  // Input tensors.
  // Names of input tensor are alias names. The mapping from aliases to real
  // input tensor names is expected to be stored as named generic signature
  // under the key "inputs" in the model export.
  // Each alias listed in a generic signature named "inputs" should be provided
  // exactly once in order to run the prediction.
  map<string, ArrayProto> inputs = 2;

  // Output filter.
  // Names specified are alias names. The mapping from aliases to real output
  // tensor names is expected to be stored as named generic signature under
  // the key "outputs" in the model export.
  // Only tensors specified here will be run/fetched and returned, with the
  // exception that when none is specified, all tensors specified in the
  // named signature will be run/fetched and returned.
  repeated string output_filter = 3;
}

// Response for PredictRequest on successful run.
message PredictResponse {
  // Output tensors.
  map<string, ArrayProto> outputs = 1;
}

// Response for current serving model info
message ServingModelInfo {
  string model_path = 1;
  // Add other info here
}
```
The user generates the class function of the corresponding language according to the above xxx.proto file, for example, xxx.pb.cc and xxx.pb.h are generated in C++, and the corresponding java files are generated in Java.

In the above code, PredictRequest is the request data structure, and PredictResponse is the response data structure returned to the user. About fields in PredictRequest and PredictResponse, we can obtain some information from **saved_model.pb/saved_model.pbtxt**, assuming that the content of **saved_model.pbtxt** is as follows:
```c
  signature_def {
    key: "serving_default"
    value {
      inputs {
        key: "input1"
        value {
          name: "input1:0"
          dtype: DT_DOUBLE
          tensor_shape {
            dim {
              size: -1
            }
          }
        }
      }
      inputs {
        key: "input2"
        value {
          name: "input2:0"
          dtype: DT_INT32
          tensor_shape {
            dim {
              size: -1
            }
          }
        }
      }
      outputs {
        key: "probabilities"
        value {
          name: "prediction:0"
          dtype: DT_FLOAT
          tensor_shape {
            dim {
              size: -1
            }
          }
        }
      }
      method_name: "tensorflow/serving/predict"
    }
  }
```
**For PredictRequest**：

1）string signature_name: User-specified signature, which can be seen from saved_model.pbtxt, here is 'serving_default'.

2）map<string, ArrayProto> inputs: Feeds, which is a map type data, key is input name, which is string type; value is the tensor (dense tensor) corresponding to this input, which is ArrayProto type, and ArrayProto is a pb array, see the definition above for details. Suppose there is an age input, and the tensor is ArrayProto t, and its value is [10], then inputs["age"] = t; if there are multiple inputs, just add them in. If there are two inputs in the above example, then the input is: {"input1:0":tensor1, "input2:0":tensor2}.

3）repeated string output_filter: Fetches, the name of the fetch tensor returned by predict is required. In the above example, there is an output, and the output_filter is: {"prediction:0"}.

**For PredictResponse**：

map<string, ArrayProto> outputs: It is a map structure, the key is the names specified in the output_filter in PredictRequest, and the value is the returned tensor.

#### Configure file
As mentioned above, the initialize function needs to be called during initialization:
```c
void* initialize(const char* model_entry, const char* model_config, int* state);
```
The second argument "**model_config**" of the initialize function is a content in json format, which has the following fields. For details, please refer to the open source code file: **serving/processor/serving/model_config.cc**
```json
{
# Enable 'session group', and set the number of sessions in the group to this value.
"session_num": 2,

# If session group is used, it indicates how each session run selects the session in the group to execute sess_run.
# The methods include:
# "MOD": According to the thread number of the request, the session num is moduloed to obtain the session num serving the current request.
# "RR": Polling to select the session in the group for service.
# default is 'RR',
"select_session_policy": "MOD",

# When using session group, true means that each session
# in the group has an independent inter/intra thread pool.
"use_per_session_threads": false,

# Users can set different cpu cores for each session in the session group,
# The format is as follows:
# "0-10;11-20" means that two sessions are bound to 0~10cores and 11~20cores respectively.
# or
# "0,1,2,3;4,5,6,7" means that two sessions are bound to 0~3cores and 4~7cores respectively.
# Different session cpu cores are separated by ';'.
"cpusets": "123;4-6",

# Users can set which physical GPUs are used by the current
# session group through options, as shown below to use GPU 0 and GPU 2.
# It should be noted that the GPU number here may not
# the same as the number from nvidia-smi.
# For example, if the user sets CUDA_VISIBLE_DEVICES=3,2,1,0,
# then the numbers 0,1,2,3 seen in deeprec correspond to physical GPUs 3,2,1,0.
# The user does not need to care about the specific GPU,
# but only needs to know the number of visible GPUs in deeprec,
# and then make corresponding settings.
"gpu_ids_list": "0,2",

# whether to use multi-stream In GPU tasks
"use_multi_stream": false,

# Whether to enable device placement optimization in GPU tasks
"enable_device_placement_optimization": false,

# Whether to execute Session run in a single thread
"enable_inline_execute": false,
  
# The default value is 4 (parameters are related to MKL performance and need to be debugged)
"omp_num_threads": 4,

# The default value is 0 (parameters are related to MKL performance and need to be debugged)
"kmp_blocktime": 0,

# Argument required to load the model, 'local' or 'redis'
# Represents loading the model into memory (local hybrid storage) and loading model parameters into redis respectively.
"feature_store_type": "local",

# [required when feature_store_type is 'redis']
"redis_url": "redis_url",

# [required when feature_store_type is 'redis']
"redis_password": "redis_password",

# [required when feature_store_type is 'redis']
# Redis read thread number
"read_thread_num": 4,

# [required when feature_store_type is 'redis'],
# Redis updating thread number   
"update_thread_num": 1,

# Default serialization uses protobuf (reserved argument)
"serialize_protocol": "protobuf",

# DeepRec inter thread num
"inter_op_parallelism_threads": 10,

# DeepRec intra thread num 
"intra_op_parallelism_threads": 10,

# Model hot update uses Session's own inter thread pool by default.              
# If the user sets this parameter, an additional model update 'inter' thread pool will be created.
# The parameter value indicates the number of threads.                           
"model_update_inter_threads": 4,

# Model hot update uses Session's own inter thread pool by default.              
# If the user sets this parameter, an additional model update 'intra' thread pool will be created.                                                                           
# The parameter value indicates the number of threads.                           
"model_update_intra_threads": 4,

# Default value 1 (reserved parameter)
"init_timeout_minutes": 1,

# signature_name, which can be obtained from saved model.pbtxt.
"signature_name": "serving_default",

# warmup file, not used means no warmup.
"warmup_file_name": "warm_up.bin",

# The storage of user model files, currently supports local/oss/hdfs
# local: "/root/a/b/c"
# oss: "oss://bucket/a/b/c"
# hdfs: "hdfs://a/b/c"
"model_store_type": "oss",

# If model_store_type is oss, then set checkpoint_dir and savedmodel_dir to oss path.
# If it is local or hdfs, then set the corresponding path.
# checkpoint_dir requires specifying the parent directory of a specific checkpoint dir,
# For example: oss://mybucket/test/ckpt_parent_dir/, then multiple versions of checkpoints
# are allowed in this directory, such as: checkpoint1/, checkpoint2/, checkpoint3/ ...
# For each version of checkpoint is the standard Tensorflow checkpoint directory structure.
# savedmodel_dir will not be updated unless the graph changes, and currently needs to be manually restarted to update.
"checkpoint_dir": "oss://mybucket/test/ckpt_parent_dir/",
"savedmodel_dir": "oss://mybucket/test/savedmodel/1616466677/",

# If oss is used, set the oss-related access id and access key below
"oss_endpoint": "oss_endpoint",
"oss_access_id": "oss_access_id",
"oss_access_key": "oss_access_key",

# If you need to print the timeline, the timeline_start_step parameter indicates
# that the timeline will start from the set number of steps
"timeline_start_step": 1,

# timeline_interval_step indicates how many steps to print a new timeline at intervals
"timeline_interval_step": 2,

# timeline_trace_count indicates how many timelines need to be collected in total
"timeline_trace_count": 3,

# timeline save location, support oss and local
# local: "/root/timeline/"
"timeline_path": "oss://mybucket/timeline/",

# EmbeddingVariable storage configuration
# 0: Use the deeperc default configuration, 1: DRAM single-level storage, 12: DRAM+SSDHASH multi-level storage
# Default value: 0
"ev_storage_type": 12,

# Set multi-level storage path, if multi-level storage is enable
"ev_storage_path": "/ssd/1/",

# The size of each level of storage in multi-level storage
"ev_storage_size": [1024, 1024]
}
```

#### Export saved_model
If the user enables the incremental_ckpt function in training, then Processor can use the incremental_ckpt to update the service during serving, thus ensuring the real-time performance of the service model.

About exporting the saved_model, users can use several different APIs, including directly using the low-level API SavedModelBuilder, or using the high-level API such as estimator. For tasks that use incremental_ckpt function, the save_incr_model switch needs to be turned on when exporting the model, so that the corresponding incremental restore subgraph can be found in the saved model.

##### SavedModelBuilder
If the user exports the saved_model by splicing the low-level APIs, it is necessary to ensure that the save_incr_model parameter is set to true (the default is false) when building the SavedModelBuilder.
```python
class SavedModelBuilder(_SavedModelBuilder):
  def __init__(self, export_dir, save_incr_model=False):
    super(SavedModelBuilder, self).__init__(export_dir=export_dir, save_incr_model=save_incr_model)

  ...
```

##### Estimator
If the user uses estimtor, the parameter save_incr_model needs to be set to True when calling estimator.export_saved_model,
```python
estimator.export_saved_model(
    export_dir_base,
    serving_input_receiver_fn,
    ...
    save_incr_model=True)
```

#### Model path configuration
In Processor, users need to provide checkpoint and saved_model paths, and processor reads meta graph information from saved_model, including signature, input, output and other information. The model parameters need to be read from the checkpoint, because the current incremental update depends on the checkpoint instead of the saved model. During the serving process, when the checkpoint is updated and the processor finds a new version of the model in the specified model directory, it will automatically load the latest model. The saved model is generally not updated unless the graph changes. If it does change, a new processor instance needs to be restarted.

The user provided files are as follows:

checkpoint:
```
/a/b/c/checkpoint_parent_dir/
     |  _ _ checkpoint_1/...
     |  _ _ checkpoint_2/...
     |  _ _ checkpoint_3/
                   |  _ _ checkpoint
                   |  _ _ graph.pbtxt
                   |  _ _ model.ckpt-0.index
                   |  _ _ model.ckpt-0.meta
                   |  _ _ model.ckpt-0.data-00000-of-00001
                   |  _ _ .incremental_ckpt/...
```
The above checkpoint_1~checkpoint_3 are a complete model directory, including full model and incremental model.

saved_model:
```
/a/b/c/saved_model/
      | _ _ saved_model.pb
      | _ _ variables
```

Taking the above as an example, in the configuration file, "checkpoint_dir" is set to "/a/b/c/checkpoint_parent_dir/", and "savedmodel_dir" is set to "/a/b/c/saved_model/"

#### Warmup
The default Model Warmup in EAS is executed when the eas task is started. For the ODL processor, because the model will be automatically updated during the serving process, the Warmup is also required for the new model, so we provide Warmup function in serving.
```
{
...
"model_config": {
  ...
  "warmup_file_name": "/xxx/xxx/warm_up.bin",
  ...
}
...
}
```
Processor currently does not support downloading warmup files, and users have two ways to provide warmup files.

1.EAS[mount oss](https://help.aliyun.com/document_detail/413364.html), example as follows,
```
"storage": [
  {
    "mount_path": "/data_oss",
    "oss": {
      "endpoint": "oss-cn-shanghai-internal.aliyuncs.com",
      "path": "oss://bucket/"
    }
  }
]
```
Mount oss://bucket/ on the local /data_oss directory, assuming that the location of the warmup file in oss is "oss://bucket/1/warmup.bin", then in the configuration file, you need to set the warmup_file_name to " /data_oss/1/warmup.bin"

2.If the user does not mount oss, another way is to use EAS to download the warmup file, the configuration is as follows,
```
{
...
"model_config": {
  ...
  "warmup_file_name": "/home/admin/docker_ml/workspace/model/warm_up.bin",
  ...
}
"warm_up_data_path": "oss://my_oss/1/warm_up.bin",
...
}
```
The warmup parameter "warm_up_data_path" of EAS needs to be configured: "oss://my_oss/1/warm_up.bin" so that the EAS framework will download this file, and the download path is "/home/admin/docker_ml/workspace/model/warm_up. bin" (different versions of eas may change, you need to consult EAS developer), the user can set the warmup_file_name to "/home/admin/docker_ml/workspace/model/warm_up.bin", so that the processor can also find the warmup_file for warmup.

## End2End example
Details see: serving/processor/tests/end2end/README
A complete end-to-end example is provided here.

## Timeline collection
By configuring timeline-related parameters in the config file, the corresponding timeline files can be obtained. These files are in binary format and cannot be displayed directly in `chrome://tracing`. Users need to go to the DeepRec directory (usually /root/ .cache/bazel/_bazel_root/) to find the config_pb2.py file which is required by gen_timeline.py script, and put it in the serving/tools/timeline directory, execute gen_timeline.py in this directory to generate the timeline that `chrome://tracing` can display.
```python
# usage:
python gen_timeline.py timeline_file my_timeline.json
```

