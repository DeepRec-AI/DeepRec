# Pipline-SmartStage

## Background

DeepRec provides the stage feature, which can realize the asynchronous execution of IO Bound operations and calculation Bound operations driven by TensorFlow runtime, improving the execution efficiency of the entire graph.

`tf.staged` requires the user to specify the boundary of the stage, on the one hand, it will increase the difficulty of use, on the other hand, the granularity of the stage will not be fine enough, making it difficult to execute more ops asynchronously. So we propose the SmartStage feature. When users do not need to have an OP-level understanding of TF Graph, they can maximize the performance of the stage.

## Feature

There is a stage in the user's original graph, by enabling the smart stage feature, it automatically optimizes the maximum possible stage range and modifies the actual physical calculation graph (without affecting the Graphdef), improving performance.


**Attention**：The prerequisite for this feature is that there is at least one stage in the user's original image

## API
ConfigProro defines the following configuration options.

- CPU scenario

```python
sess_config = tf.ConfigProto()
sess_config.graph_options.optimizer_options.do_smart_stage = True
```
- GPU scenario

```python
sess_config = tf.ConfigProto()
sess_config.graph_options.optimizer_options.do_smart_stage_gpu = True
```

## Example

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
# For GPU training, consider using the following options instead of do_smart_stage for better performance
# config.graph_options.optimizer_options.do_smart_stage_gpu = True
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
| DLRM | w/o smart stage |  212 (+ 1.05x)   |

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
