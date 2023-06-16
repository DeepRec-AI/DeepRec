# 流水线-Stage

## 背景

在一个通常的TensorFlow训练任务中通常由样本数据的读取，图计算构成，样本数据的读取属于IO bound操作，在整个E2E的耗时占据表较大的百分比，从而拖慢训练任务，同时并不能高效的利用计算资源（CPU、GPU）。DeepRec已经提供了stage 功能，它的思想来源于TensorFlow的StagingArea功能，我们在DeepRec提供了API `tf.staged`，用户通过显式的指定图中哪一部分需要stage，以及在Session创建的时候加入`tf.make_prefetch_hook()`在TensorFlow runtime驱动异步执行，从而提高整张图的执行效率。

## 用户接口

`tf.staged`，对输入的 `features` 进行预取，返回预取后的 tensor。

| 参数                            | 含义                                                                                                                                        | 默认                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| features                       | 需要异步化执行的 op，可以是 tensor、list of tensor（list 中每一个元素都是 tensor ） 或者 dict of tensor （dict 中的 key 都是 string，value 都是 tensor）| 必选参数                                                                  |
| feed_dict                      | 将stage子图元素映射为对应tensor的字典                                                                                                           | {}                                                                      |
| capacity                       | 缓存的 `items`异步化执行结果的最大个数                                                                                                           | 1                                                                       |
| num_threads                    | 异步化执行 `items`的线程数                                                                                                                     | 1                                                                       |
| num_clients                    | 消耗预取结果的消费者数量                                                                                                                        | 1                                                                       |
| timeout_millis                 | 预取结果等待缓存区可用的最大等待时间，超时后本次预取结果将会被丢弃                                                                                      | 300000 ms                                                               |
| closed_exception_types         | 被识别为正常退出的异常类型                                                                                                                      | (`tf.errors.OUT_OF_RANGE`,)                                              |
| ignored_exception_types        | 被识别可忽略跳过的异常类型                                                                                                                      | ()                                                                       |
| use_stage_subgraph_thread_pool | 是否在独立线程池上运行Stage子图，需要先创建独立线程池                                                                                               | False(若为True则必须先创建独立线程池)                                         |
| stage_subgraph_thread_pool_id  | 如果开启了在独立线程池上运行Stage子图，用于指定独立线程池索引，需要先创建独立线程池，并打开use_stage_subgraph_thread_pool选项                               | 0，索引范围为[0, 创建的独立线程池数量-1]                                       |
| stage_subgraph_stream_id       | GPU Multi-Stream 场景下, stage子图执行使用的gpu stream的索引                                                                                    | 0 (0表示stage子图共享计算主图使用的gpu stream, 索引范围为[0, gpu stream总数-1]) |
| name                           | 预取操作的名称                                                                                                                                | None (表示自动生成)                                                        |

Session中加入`tf.make_prefetch_hook()`hook

```python
hooks=[tf.make_prefetch_hook()]
with tf.train.MonitoredTrainingSession(hooks=hooks, config=sess_config) as sess:
```

- 创建独立线程池（可选）

```python
sess_config = tf.ConfigProto()
sess_config.session_stage_subgraph_thread_pool.add() # 增加一个独立线程池
sess_config.session_stage_subgraph_thread_pool[0].inter_op_threads_num = 8 # 独立线程池中inter线程数量
sess_config.session_stage_subgraph_thread_pool[0].intra_op_threads_num = 8 # 独立线程池中intra线程数量
sess_config.session_stage_subgraph_thread_pool[0].global_name = "StageThreadPool_1" # 独立线程池名称
```

- GPU Multi-Stream Stage(可选)
```python
sess_config = tf.ConfigProto()
sess_config.graph_options.rewrite_options.use_multi_stream = (rewriter_config_pb2.RewriterConfig.ON) # 开启GPU Multi-Stream功能
sess_config.graph_options.rewrite_options.multi_stream_opts.multi_stream_num = 2 # 设定可用的stream数量, 其中0号stream提供给计算主图使用
sess_config.graph_options.optimizer_options.stage_multi_stream = True # 开启GPU Multi-Stream Stage
```
> GPU Multi-Stream Stage功能还可以通过开启GPU MPS来实现更好的性能, 请见[GPU-MultiStream](./GPU-MultiStream.md).

**注意事项**

- 待异步化的计算应该尽可能少和后续主体计算争抢资源（gpu、cpu、线程池等）
- `capacity` 更大会消耗更多的内存或显存，同时可能会抢占后续模型训练的 CPU 资源，建议设置为后续计算时间/待异步化时间。可以从 1 开始逐渐向上调整
- `num_threads` 并不是越大越好，只需要可以让计算和预处理重叠起来即可，数量更大会抢占模型训练的 CPU 资源。计算公式：num_threads >= 预处理时间 / 训练时间，可以从 1 开始向上调整
- `tf.make_prefetch_hook()`一定要加上，否则会hang住
- 

## 代码示例

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

# mark target 节点
tf.train.mark_target_node([target])

with tf.train.MonitoredTrainingSession(hooks=[tf.make_prefetch_hook()]) as sess:
  for i in range(5):
      print(sess.run([target]))
```

