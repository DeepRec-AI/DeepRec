# 自动流水线-SmartStage
## 背景
DeepRec已经提供了stage 功能，该功能可以实现IO Bound操作和计算Bound操作在TensorFlow runtime的驱动下异步执行，从而提高整张图的执行效率。

由于`tf.staged`需要用户指定stage的边界，一方面会增加使用难度，另一方面会导致stage颗粒度不够精细，难以做到更多op的异步执行。因此我们提出了SmartStage功能。用户不需要对TF Graph有OP级别理解的情况下，就可以使stage发挥最大的性能提升。

## 功能说明
通过开启smart stage功能，自动化的寻优最大可以stage的范围，修改实际物理计算图（不影响Graphdef图），从而提高性能。

## 用户接口
### 1. 自动SmartStage(推荐)
自动SmartStage的前提是模型使用了`tf.data.Iterator`接口从`tf.data.Dataset`中读取样本数据。

1. `tf.SmartStageOptions`接口返回执行stage子图的配置，其参数如下：

| 参数                            | 含义                                                                                                                                        | 默认                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| capacity                       | 缓存的异步化执行结果的最大个数                                                                                                                   | 1                                                                       |
| num_threads                    | 异步化执行stage子图的线程数                                                                                                                     | 1                                                                       |
| num_clients                    | 消耗预取结果的消费者数量                                                                                                                        | 1                                                                       |
| timeout_millis                 | 预取结果等待缓存区可用的最大等待时间，超时后本次预取结果将会被丢弃                                                                                      | 300000 ms                                                               |
| closed_exception_types         | 被识别为正常退出的异常类型                                                                                                                      | (`tf.errors.OUT_OF_RANGE`,)                                              |
| ignored_exception_types        | 被识别可忽略跳过的异常类型                                                                                                                      | ()                                                                       |
| use_stage_subgraph_thread_pool | 是否在独立线程池上运行Stage子图，需要先创建独立线程池                                                                                               | False(若为True则必须先创建独立线程池)                                         |
| stage_subgraph_thread_pool_id  | 如果开启了在独立线程池上运行Stage子图，用于指定独立线程池索引，需要先创建独立线程池，并打开use_stage_subgraph_thread_pool选项                               | 0，索引范围为[0, 创建的独立线程池数量-1]                                       |
| stage_subgraph_stream_id       | GPU Multi-Stream 场景下, stage子图执行使用的gpu stream的索引                                                                                    | 0 (0表示stage子图共享计算主图使用的gpu stream, 索引范围为[0, gpu stream总数-1]) |
| graph                          | 需要执行SmartStage优化的Graph，需要与传递给Session的Graph相同                                                                                     | None (表示使用默认Graph)                                                   |
| name                           | 预取操作的名称                                                                                                                                | None (表示自动生成)                                                        |
    
> 关于如何创建独立线程池以及如何使用GPU Multi-Stream，请参见[流水线](./Stage.md)。

2. `tf.SmartStageOptions`接口生成的配置需要赋值给`tf.ConfigProto`。
    ```python
    sess_config = tf.ConfigProto()
    smart_stage_options = tf.SmartStageOptions(capacity=40, num_threads=4)
    sess_config.graph_options.optimizer_options.smart_stage_options.CopyFrom(smart_stage_options)
    ```
3. 设置`tf.ConfigProto`中的如下选项来开启SmartStage。
    - CPU场景
    ```python
    sess_config = tf.ConfigProto()
    sess_config.graph_options.optimizer_options.do_smart_stage = True # 开启SmartStage
    ```

    - GPU场景
    ```python
    sess_config = tf.ConfigProto()
    sess_config.graph_options.optimizer_options.do_smart_stage = True # 开启SmartStage
    sess_config.graph_options.optimizer_options.stage_subgraph_on_cpu = True # 针对GPU场景优化的选项
    ```

4. Session中加入`tf.make_prefetch_hook()` hook

### 2. 图中存在Stage阶段时的SmartStage
原图已经使用`tf.staged`接口手动分图。
> 关于`tf.staged`接口请参见[流水线](./Stage.md)。

1. 直接设置`tf.ConfigProto`中的相关选项即可开启SmartStage。
    **CPU场景**
    ```python
    sess_config = tf.ConfigProto()
    sess_config.graph_options.optimizer_options.do_smart_stage = True # 开启SmartStage
    ```

    **GPU场景**
    ```python
    sess_config = tf.ConfigProto()
    sess_config.graph_options.optimizer_options.do_smart_stage = True # 开启SmartStage
    sess_config.graph_options.optimizer_options.stage_subgraph_on_cpu = True # 针对GPU场景优化的选项
    ```

2. Session中加入`tf.make_prefetch_hook()` hook

## 代码示例
### 自动SmartStage(推荐)

```python
import tensorflow as tf

def parse_csv(value):
    v = tf.io.decode_csv(value, record_defaults=[[''], ['']])
    return v
    
dataset = tf.data.TextLineDataset('./test_data.csv')
dataset = dataset.batch(2)
dataset = dataset.map(parse_csv, num_parallel_calls=2)
dataset_output_types = tf.data.get_output_types(dataset)
dataset_output_shapes = tf.data.get_output_shapes(dataset)
iterator = tf.data.Iterator.from_structure(dataset_output_types, dataset_output_shapes)
xx = iterator.get_next()
xx = list(xx)

init_op = iterator.make_initializer(dataset)

var = tf.get_variable("var", shape=[100, 3], initializer=tf.ones_initializer())
xx[0] = tf.string_to_hash_bucket(xx[0], num_buckets=10)
xx[0] = tf.nn.embedding_lookup(var, xx[0])
xx[1]=tf.concat([xx[1], ['xxx']], axis = 0)
target = tf.concat([tf.as_string(xx[0]), [xx[1], xx[1]]], 0)

config = tf.ConfigProto()
# enable smart stage
config.graph_options.optimizer_options.do_smart_stage = True
smart_stage_options = tf.SmartStageOptions(capacity=1, num_threads=1)
config.graph_options.optimizer_options.smart_stage_options.CopyFrom(smart_stage_options)

# 对于GPU训练，可以考虑开启以下选项来获得更好的性能
# config.graph_options.optimizer_options.stage_subgraph_on_cpu = True
    
# mark target 节点
tf.train.mark_target_node([target])

scaffold = tf.train.Scaffold(
    local_init_op=tf.group(tf.local_variables_initializer(), init_op))
with tf.train.MonitoredTrainingSession(config=config, scaffold=scaffold, 
                                       hooks=[tf.make_prefetch_hook()]) as sess:
    for i in range(5):
        print(sess.run([target]))
```

### 图中存在Stage阶段时的SmartStage
```python
import tensorflow as tf

filename_queue = tf.train.string_input_producer(['1.txt'])
reader = tf.TextLineReader()
k, v = reader.read(filename_queue)

var = tf.get_variable("var", shape=[100, 3], initializer=tf.ones_initializer())
v = tf.train.batch([v], batch_size=2, capacity=20 * 3)
v0, v1 = tf.decode_csv(v, record_defaults=[[''], ['']], field_delim=',')
xx = tf.staged([v0, v1])

xx[0]=tf.string_to_hash_bucket(xx[0],num_buckets=10)
xx[0] = tf.nn.embedding_lookup(var, xx[0])
xx[1]=tf.concat([xx[1], ['xxx']], axis = 0)
target = tf.concat([tf.as_string(xx[0]), [xx[1], xx[1]]], 0)

config = tf.ConfigProto()
# enable smart stage
config.graph_options.optimizer_options.do_smart_stage = True
# 对于GPU训练，可以考虑开启以下选项来获得更好的性能
# config.graph_options.optimizer_options.stage_subgraph_on_cpu = True

# mark target 节点
tf.train.mark_target_node([target])

with tf.train.MonitoredTrainingSession(config=config,
                                       hooks=[tf.make_prefetch_hook()]) as sess:
for i in range(5):
    print(sess.run([target]))
```

## 性能对比
### CPU场景
在modelzoo中的DLRM模型中测试该功能
机型为Aliyun ECS 实例 ecs.hfg7.8xlarge

- Model name: Intel(R) Xeon(R) Platinum 8369HC CPU @ 3.30GHz
- CPU(s): 32
- Socket(s): 1
- Core(s) per socket: 16
- Thread(s) per core: 2
- Memory: 128G

|      |      case       | global steps/sec |
| :--: | :-------------: | :--------------: |
| DLRM | w/o smart stage |  201 (baseline)  |
| DLRM | w/  smart stage |  212 (+ 1.05x)   |

### GPU场景

在modelzoo中的模型测试该功能在GPU训练场景下的性能。

机器配置：

| CPU  | Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz | 64核心 |
| ---- | --------------------------------------------- | ------ |
| GPU  | NVIDIA A100 80G                               | 单卡   |
| MEM  | 492G                                          |        |

性能结果对比：

| 模型   | 不开启SmartStage <br>(global steps/sec) | do_smartstage <br>(global steps/sec) | do_smartstage_gpu <br>(global steps/sec) |
| ------ | --------------------------------------- | ------------------------------------ | ---------------------------------------- |
| DIEN   | 17.1673                                 | 16.918                               | 17.2557                                  |
| DIN    | 137.584                                 | 132.619                              | 165.069                                  |
| DLRM   | 91.6982                                 | 67.735                               | 188.105                                  |
| DSSM   | 92.4544                                 | 83.7194                              | 101.352                                  |
| DeepFM | 74.7011                                 | 62.1227                              | 93.0858                                  |
