# Pipline-SmartStage

## Background

DeepRec provides the stage feature, which can realize the asynchronous execution of IO Bound operations and calculation Bound operations driven by TensorFlow runtime, improving the execution efficiency of the entire graph.

`tf.staged` requires the user to specify the boundary of the stage, on the one hand, it will increase the difficulty of use, on the other hand, the granularity of the stage will not be fine enough, making it difficult to execute more ops asynchronously. So we propose the SmartStage feature. When users do not need to have an OP-level understanding of TF Graph, they can maximize the performance of the stage.

## Feature

By enabling the smart stage feature, it automatically optimizes the maximum possible stage range from a certain starting node and modifies the actual physical calculation graph (without affecting the Graphdef), improving performance.

## API
### 1. Automatically SmartStage (Recommend)
The premise of automatic SmartStage is that the model uses the `tf.data.Iterator` interface to read sample data from `tf.data.Dataset`.

1. The `tf.SmartStageOptions` interface returns the configuration for executing the stage subgraph, and its parameters are as follows:

| parameter                      | description                                                                                                                                                                                                                     | default value                                                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| capacity                       | The maximum number of cached asynchronous execution results.                                                                                                                                                                    | 1                                                                                                                                        |
| num_threads                    | Number of threads to execute stage subgraph asynchronously.                                                                                                                                                                     | 1                                                                                                                                        |
| num_client                     | Number of clients of prefetched sample.                                                                                                                                                                                         | 1                                                                                                                                        |
| timeout_millis                 | Max milliseconds put op can take.                                                                                                                                                                                               | 300000 ms                                                                                                                                |
| closed_exception_types         | Exception types recognized as graceful exits.                                                                                                                                                                                   | (`tf.errors.OUT_OF_RANGE`,)                                                                                                              |
| ignored_exception_types        | Exception types that are recognized to be ignored and skipped.                                                                                                                                                                  | ()                                                                                                                                       |
| use_stage_subgraph_thread_pool | Whether to run the Stage subgraph on an independent thread pool, you need to create an independent thread pool first.                                                                                                           | False (If it is True, a separate thread pool must be created first)                                                                      |
| stage_subgraph_thread_pool_id  | If you enable the stage subgraph to run on the independent thread pool to specify the independent thread pool index, you need to create an independent thread pool first, and enable the use_stage_subgraph_thread_pool option. | 0, The index range is [0, the number of independent thread pools created - 1]                                                            |
| stage_subgraph_stream_id       | In the GPU Multi-Stream scenario, the index of gpu stream used by stage subgraph.                                                                                                                                               | 0 (0 means that the stage subgraph shares the gpu stream used by the main graph, the index range is [0, total number of GPU streams -1]) |
| graph                          | The Graph that needs to be optimized by SmartStage, which is the same as the Graph passed to the Session                                                                                                                        | None (Use default graph)                                                                                                                 |
| name                           | Name of prefetching operations.                                                                                                                                                                                                 | None (Automatic generated)                                                                                                               |

> For how to create an independent thread pool or use GPU Multi-Stream, please refer to [Pipeline-Stage](./Stage.md).

2. The configuration generated by the `tf.SmartStageOptions` interface needs to be assigned to `tf.ConfigProto`.
    ```python
    sess_config = tf.ConfigProto()
    smart_stage_options = tf.SmartStageOptions(capacity=40, num_threads=4)
    sess_config.graph_options.optimizer_options.smart_stage_options.CopyFrom(smart_stage_options)
    ```

3. Set the following options in `tf.ConfigProto` to enable SmartStage.
    - CPU scenario
    ```python
    sess_config = tf.ConfigProto()
    sess_config.graph_options.optimizer_options.do_smart_stage = True
    ```
    - GPU scenario
    ```python
    sess_config = tf.ConfigProto()
    sess_config.graph_options.optimizer_options.do_smart_stage = True
    sess_config.graph_options.optimizer_options.stage_subgraph_on_cpu = True
    ```
4. Add `tf.make_prefetch_hook()` hook to Session.

### 2. SmartStage when Graph contains Stage
The original graph has been manually split using the `tf.staged` interface.
> For more detail of `tf.staged`, please refer to [Pipeline-Stage](./Stage.md).

1. Set the following options in `tf.ConfigProto` to enable SmartStage.
    - CPU scenario
    ```python
    sess_config = tf.ConfigProto()
    sess_config.graph_options.optimizer_options.do_smart_stage = True
    ```
    - GPU scenario
    ```python
    sess_config = tf.ConfigProto()
    sess_config.graph_options.optimizer_options.do_smart_stage = True
    sess_config.graph_options.optimizer_options.stage_subgraph_on_cpu = True
    ```
2. Add `tf.make_prefetch_hook()` hook to Session.

## Example
#### Automatically SmartStage (Recommend)
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

# For GPU training, consider enabling the following options for better performance
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

#### SmartStage when Graph contains Stage.
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
# For GPU training, consider enabling the following options for better performance
# config.graph_options.optimizer_options.stage_subgraph_on_cpu = True

# mark target node
tf.train.mark_target_node([target])

with tf.train.MonitoredTrainingSession(config=config,
                                       hooks=[tf.make_prefetch_hook()]) as sess:
  for i in range(5):
    print(sess.run([target]))
```
## Performance

- CPU scenario

The performance of this feature in the DLRM model in modelzoo.

The Aliyun ECS instance is ecs.hfg7.8xlarge

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

- GPU scenario

The performance of this feature in the GPU training scenario in the model in modelzoo.

machine configuration:

| Resource | Description | Cores |
| ---- | --------------------------------------------- | ------ |
| CPU  | Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz | 64 core |
| GPU  | NVIDIA A100 80G                               | 1 gpu   |
| MEM  | 492G                                          |        |

Performance results comparison:

| model   | w/o SmartStage <br>(global steps/sec) | do_smartstage <br>(global steps/sec) | do_smartstage_gpu <br>(global steps/sec) |
| ------ | --------------------------------------- | ------------------------------------ | ---------------------------------------- |
| DIEN   | 17.1673                                 | 16.918                               | 17.2557                                  |
| DIN    | 137.584                                 | 132.619                              | 165.069                                  |
| DLRM   | 91.6982                                 | 67.735                               | 188.105                                  |
| DSSM   | 92.4544                                 | 83.7194                              | 101.352                                  |
| DeepFM | 74.7011                                 | 62.1227                              | 93.0858                                  |
