# Stage

## Background

A TensorFlow training task is usually composed of sample data reading and graph calculation. The reading of sample data is an IO bound operation, which occupies a large percentage of the entire E2E time, then slowing down the training task. It cannot efficiently use computing resources (CPU, GPU). DeepRec has already provided the stage function. Its idea comes from the StagingArea function of TensorFlow. We provide the API `tf.staged` in DeepRec. The user explicitly specifies which part of the graph needs the stage, and adds `tf.make_prefetch_hook()` when the Session is created. Under the asynchronous execution of TensorFlow runtime, the execution efficiency of the entire graph is improved.

## API

`tf.staged`, prefetch the input `features`, and return the prefetched tensor.

| parameter                    | description                                                         | default value                                                |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| features                | The tensor that needs to be executed asynchronously: tensor, list of tensor (each element in the list is a tensor) or dict of tensor (the keys in the dict are all strings, and the values are all tensors) | required                                               |
| capacity                | The maximum number of cached `items` asynchronous execution results.                    | 1                                                      |
| num_threads             | Number of threads to execute `items` asynchronously.                                 | 1                                                      |
| items                   | A list of feed_dict keys that `items` depends on                        | None, `items` does not depend on feed_dict                       |
| feed_generator          | A generator object for the value of the feed_dict that `items` depends on. A generator object in Python is a method that yields a list. Through this generator object, users can use pure Python for flexible data preprocessing, similar to tensor_pack, see examples for interface and usage. | None, `features` does not depend on feed_dict                    |
| closed_exception_types  | Exception types recognized as graceful exits                                  | (`tf.errors.OutOfRangeError`, `errors.CancelledError`) |
| ignored_exception_types | Exception types that are recognized to be ignored and skipped                                   | ()                                                     |
| use_stage_subgraph_thread_pool   | Whether to run the Stage subgraph on an independent thread pool, you need to create an independent thread pool first        | False(Optional, if it is True, a separate thread pool must be created first)            |
| stage_subgraph_thread_id         | If you enable the stage subgraph to run on the independent thread pool to specify the independent thread pool index, you need to create an independent thread pool first, and enable the use_stage_subgraph_thread_pool option. | 0, The index range is [0, the number of independent thread pools created - 1]               |

Adds `tf.make_prefetch_hook()`hook when create session.

```python
hooks=[tf.make_prefetch_hook()]
with tf.train.MonitoredTrainingSession(hooks=hooks, config=sess_config) as sess:
```

Create a separate thread pool (optional)

```python
sess_config = tf.ConfigProto()
sess_config.session_stage_subgraph_thread_pool.add() # Add a thread pool
sess_config.session_stage_subgraph_thread_pool[0].inter_op_threads_num = 8 # inter thread number in thread pool
sess_config.session_stage_subgraph_thread_pool[0].intra_op_threads_num = 8 # intra thread number in thread pool
sess_config.session_stage_subgraph_thread_pool[0].global_name = "StageThreadPool_1" # thread pool name
```

attention:

- Computations to be asynchronous should compete with subsequent main computations for resources as little as possible (gpu, cpu, thread pool, etc.)
- A larger `capacity` will consume more memory or video memory, and may occupy CPU resources for subsequent model training. It is recommended to set it to follow-up calculation time/waiting for asynchronization time. It can be adjusted gradually upwards starting from 1.
- `num_threads` is not as big as possible, it just needs to allow calculation and preprocessing to overlap, and a larger number will preempt CPU resources for model training. Calculation formula: num_threads >= preprocessing time / training time, can be adjusted upwards from 1.
- `tf.make_prefetch_hook()` must be added, otherwise it will hang.

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

# mark target node
tf.train.mark_target_node([target])

with tf.train.MonitoredTrainingSession(hooks=[tf.make_prefetch_hook()]) as sess:
  for i in range(5):
      print(sess.run([target]))
```
