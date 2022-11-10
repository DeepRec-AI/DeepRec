# Processor

## 介绍
DeepRec Serving Processor是用于线上高性能服务的Library，它以DeepRec为基石，实现多种实用功能：

● 支持自动发现并导入全量模型；

● 支持增量更新，减少加载模型带来的时间消耗，使模型上线更实时，更轻便；

● 支持异步更新模型，减少线上serving性能抖动；

● 支持模型回退；

● 支持启动WarmUp，避免首次加载模型慢启动问题；

● 支持模型本地存储(内存+Pmem+SSD)以及分布式存储服务(多节点shard存储模型参数)等。

● 同时也兼容原生Tensorflow，对于使用原生Tensorflow训练出来的graph也能正常serving。

● 简易的API接口，让用户对接更方便。

Processor的产出是一个独立的so，用户可以很方便的对接到自己的Serving RPC框架中。

## 编译
编译详见[https://github.com/alibaba/DeepRec](https://github.com/alibaba/DeepRec)项目首页“**How to Build serving library**”部分，编译的产出是“**libserving_processor.so**”。
## 使用
用户有两种使用方式：

第一，在用户框架代码中直接[dlopen](https://linux.die.net/man/3/dlopen)从而加载so中的符号。

第二，可以结合头文件“**serving/processor/serving/processor.h**”使用，头文件中将Processor相关的API暴露了，通过头文件和“**libserving_processor.so**”来调用serving API也比较方便。

**需要注意**：如果不是使用DeepRec docker，那么可能需要一些额外的so依赖，包括：libiomp5.so，libmklml_intel.so，libstdc++.so.6，用户可以[直接下载](http://tfsmoke1.cn-hangzhou.oss.aliyun-inc.com/deeprec/serving_processor_so.tar.gz)，然后在执行时候Preload这些so。

#### API接口
Processor提供以下几组C API接口，用户在自己的Serving框架中需要调用下列接口。

**1) initialize**
```c
void* initialize(const char* model_entry, const char* model_config, int* state);
```
**参数：**

model_entry: 默认传字符串""(注意不是NULL)。

model_config：从配置文件中读取的json内容。

state：返回给用户Serving框架的状态，0为正常，-1为异常。

**返回值：**
返回一个指针，即下面process函数中的model_buf参数。每次调用process需要将此参数传入。

**使用方式：**
在Serving RPC框架启动时候调用一次initialize函数，将返回的指针保存起来，并且在后续调用process需要将此参数传入。

**用户使用：**
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
**参数：**

model_buf：initialize的返回值。

input_data：用户请求request被protobuf序列化成的字节流数据，protobuf格式见下面PredictRequest “数据格式”。

input_size：输入request的大小。

output_data：预测返回的结果，是protobuf格式序列化后的字节流，protobuf格式见下面PredictResponse “数据格式”。(注意：返回的buffer是分配在堆内存上的，用户框架需要负责回收内存，否则会内存泄漏。)

output_size：输出response的大小。

**返回值：**
返回服务码，200代表OK，500代表服务出错。

**使用方式：**
用户Serving框架接收到Client发送的请求，如果请求的数据格式是Processor需要的格式(见下面“数据格式”)，直接调用Process函数执行预测即可。如果数据格式不一致，那么需要转成Processor需要的格式(见下面“数据格式”)之后调用Process函数。

**用户使用：**
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
**参数：**

model_buf：initialize的返回值。

output_data：当前serving model信息的返回结果，是protobuf格式序列化后的字节流，protobuf格式见下面ServingModelInfo “数据格式”。(注意：返回的buffer是分配在堆内存上的，用户框架需要负责回收内存，否则会内存泄漏。)

output_size：输出output_data的大小。

**返回值：**
返回服务码，200代表OK，500代表服务出错。

**使用方式：**
用户在需要确认当前正在服务的模型信息时可以调用此API查询。

**用户使用：**
```c
void* model = initialize(xx, xx, xx);
...
void* output_data = nullptr;
int output_size = 0;
int state = get_serving_model_info(model, &output_data, &output_size);
```

#### 数据格式
Processor需要的Request，Response等数据格式如下所示，这里我们使用Protobuf作为数据存储格式。同DeepRec下“**serving/processor/serving/predict.proto**”一致。
Protobuf是Protocol Buffers的简称，它是一种数据描述语言，用于描述一种轻便高效的结构化数据存储格式。 Protobuf可以用于结构化数据串行化，或者说序列化。简单来说，在不同的语言之间，数据可以相互解析。Java的protobuf数据被序列化之后在c++中可以原样解析出来，这样方便支持各种场景下的数据互通。
用户在客户端需要将数据封装成下面格式，或者在Process之前转成此格式。
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
用户根据上述xxx.proto文件生成对应语言的类函数，譬如在C++中生成xxx.pb.cc和xxx.pb.h，在Java中生成对应的java文件。生成的文件中就是我们熟悉的C++的结构体，然后对相应的字段做赋值即可。

上述代码中PredictRequest是请求request数据结构，PredictResponse是返回给用户的response数据结构。这些参数的设置需要从**saved_model.pb/saved_model.pbtxt**中获取一些信息，假设**saved_model.pbtxt**内容如下所示：
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
**对于 PredictRequest**：

1）string signature_name：用户指定的签名，这可以从saved_model.pbtxt中看到，即：serving_default

2）map<string, ArrayProto> inputs：即feeds，这是一个map类型数据，key是input name，是string类型；value是这个input对应的tensor(是dense tensor)，是ArrayProto类型，ArrayProto是一个pb数组，具体见上面定义。假设现在有一个age输入，同时tensor是ArrayProto t，其值是[10]，那么inputs["age"] = t; 如果有多个input，都增加进去即可。如果上示例中有有两个input，那么输入为: {"input1:0":tensor1, "input2:0":tensor2}。

3）repeated string output_filter：即fetches，需要predict返回的fetch tensor的name，如果上示例中有有一个output，那么output_filter为: {"prediction:0"}。

**对于 PredictResponse**：

map<string, ArrayProto> outputs：是map结构，key是 PredictRequest中output_filter中指定的names，value是返回的tensor。
#### 配置文件
上面提到，在初始化时候需要调用函数：
```c
void* initialize(const char* model_entry, const char* model_config, int* state);
```
函数的第二个参数“**model_config**”是一个json格式的配置，有如下字段，也可以参考开源代码文件：**serving/processor/serving/model_config.cc**
```json
{
# 使用session group，并且设置group中session数量为此值
"session_num": 2,

# 若使用session group，表示每次session run以何种方式
# 选择group中的session来执行sess_run。方式包括：
# "MOD": 根据request所在线程号对session num
#        取模获取为当前request服务的session num。
# "RR": 轮询选择group中的session进行服务。
"select_session_policy": "MOD",

# 在使用session group任务中，true表示group
# 中每个session都拥有独立的inter/intra线程池。
"use_per_session_threads": false,

# 是否单线程执行 Session run
"enable_inline_execute": false
  
# 默认值(参数和MKL性能有关，需要调试)
"omp_num_threads": 4,

# 默认值(参数和MKL性能有关，需要调试)
"kmp_blocktime": 0,

# 加载模型参数，'local' 或者 'redis'
# 分别代表加载模型到内存(本地混合存储) 和 加载模型参数到redis中
"feature_store_type": "local",

# [feature_store_type是'redis'需要]
"redis_url": "redis_url",

# [feature_store_type是'redis'需要]
"redis_password": "redis_password",

# [feature_store_type是'redis'需要], redis读线程数
"read_thread_num": 4,
# [feature_store_type是'redis'需要]，redis更新模型线程数 "update_thread_num": 1,

# 默认序列化使用protobuf(预留参数)
"serialize_protocol": "protobuf",

# DeepRec中inter线程数
"inter_op_parallelism_threads": 10,

# DeepRec中intra线程数
"intra_op_parallelism_threads": 10,

# 用户设置模型更新过程中使用的inter线程数，
# 若设置此参数则会创建额外inter模型更新线程池。
# 模型更新默认使用Session自有inter线程池。
"model_update_inter_threads": 4,

# 用户设置模型更新过程中使用的intra线程数，
# 若设置此参数则会创建额外intra模型更新线程池。
# 模型更新默认使用Session自有intra线程池。
"model_update_intra_threads": 4,

# 默认值(预留参数)
"init_timeout_minutes": 1,

# 从saved model中获取的signature_name
"signature_name": "serving_default",

# warmup文件，是request protobuf被序列化之后的文件，
# 用于预热服务。(可以为空，表示不预热)
"warmup_file_name": "warm_up.bin",

# 用户模型文件的存储位置，目前支持local/oss/hdfs
# local: "/root/a/b/c"
# oss: "oss://bucket/a/b/c"
# hdfs: "hdfs://a/b/c"
"model_store_type": "oss",

# model_store_type是oss，那么如下设置checkpoint_dir和savedmodel_dir为oss地址
# 如果是local或者hdfs，那么设置对应地址即可。
# checkpoint_dir要求指定具体的checkpoint dir的父目录，
# 例如：oss://mybucket/test/ckpt_parent_dir/，那么在此目录下允许存在
# 多个版本的checkpoint，如：checkpoint1/，checkpoint2/，checkpoint3/ ...
# 对于每个版本的checkpoint的都是标准的Tensorflow checkpoint目录结构。
# savedmodel_dir不会更新，除非graph改变了，目前需要手动重启更新。
"checkpoint_dir": "oss://mybucket/test/ckpt_parent_dir/",
"savedmodel_dir": "oss://mybucket/test/savedmodel/1616466677/",

# [如果上面使用了oss], 下面设置oss相关的access id和access key
"oss_endpoint": "oss_endpoint",
"oss_access_id": "oss_access_id",
"oss_access_key": "oss_access_key",

# [如果需要打印timeline]，增加下面参数
# 从timeline_start_step步开始打timeline
"timeline_start_step": 1,

# 间隔timeline_interval_step步打印一个新的timeline
"timeline_interval_step": 2,

# 共采集timeline_trace_count个timeline
"timeline_trace_count": 3,

# timeline保存位置，支持oss和local
# local: "/root/timeline/"
"timeline_path": "oss://mybucket/timeline/",

# EmbeddingVariable 存储配置
# 0: 使用原图配置, 1: DRAM单级存储, 12: DRAM+SSDHASH 多级存储
# 默认值: 0
"ev_storage_type": 12,

# 多级存储路径, 如果设置了多级存储
"ev_storage_path": "/ssd/1/",

# 多级存储中每级存储的大小
"ev_storage_size": [1024, 1024]
}
```
#### 模型路径配置
在Processor中，用户需要提供checkpoint以及saved_model，processor从saved_model中读取meta graph 信息，包括signature，input，output等信息。模型参数需要从checkpoint中读取，原因是现在的增量更新依赖checkpoint，而不是saved model。在serving过程中，当checkpoint更新了，processor在指定的模型目录下发现有新的版本的模型，会自动加载最新的模型。saved model一般不会更新，除非graph变化，当真的变化了，需要重启新的processor instance。

用户提供文件如下：

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
上述checkpoint_1～checkpoint_3分别 是一个完整的模型目录，包含全量模型以及增量模型。

saved_model:
```
/a/b/c/saved_model/
      | _ _ saved_model.pb
      | _ _ variables
```
以上述为例，在配置文件中"checkpoint_dir"设置为“/a/b/c/checkpoint_parent_dir/”，"savedmodel_dir"设置为“/a/b/c/saved_model/”

#### Warmup
EAS中的Warmup是在eas任务启动的时候执行的，对于ODL processor来说，因为在serving过程中也会自动更新模型，所以对于新的模型也需要进行Warmup，所以我们在ODL Processor中提供了Warmup模型的功能。
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
Processor目前不支持下载warmup文件，用户有两种方式来进行提供warmup文件。

1.EAS[挂载oss](https://help.aliyun.com/document_detail/413364.html)，示例如下，
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
将oss://bucket/挂载在本地的/data_oss目录上，假设warmup文件在oss中位置是"oss://bucket/1/warmup.bin"，那么在配置文件中需要将warmup_file_name设置成"/data_oss/1/warmup.bin"。

2.如果用户不挂载oss，另一种方式是借助EAS下载的warmup文件，配置如下，
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
首先需要配置EAS的warmup参数"warm_up_data_path": "oss://my_oss/1/warm_up.bin",这样EAS框架会下载这个文件，下载的路径是"/home/admin/docker_ml/workspace/model/warm_up.bin"(不同版本eas可能会变化，需要咨询eas同学)，用户可以设置warmup_file_name为"/home/admin/docker_ml/workspace/model/warm_up.bin"，这样processor也可以找到并且进行后续的warmup。

## 示例
End2End的示例详见：serving/processor/tests/end2end/README
这里提供了一个完整的端到端的示例。

## Timeline收集
通过在config中配置timeline相关参数，能够获取对应的timeline文件，这些文件是二进制格式，不能直接在`chrome://tracing`中进行展示，用户需要在编译DeepRec完成后目录(一般是/root/.cache/bazel/_bazel_root/)下find这个config_pb2.py文件，并放在serving/tools/timeline目录下，在此目录下执行生成`chrome://tracing`能展示的timeline。
