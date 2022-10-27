# Incremental Checkpoint
## 功能说明
在大规模稀疏场景中，数据存在冷热倾斜，相邻全量checkpoint中大部分数据都是没有任何变化的。在这个背景下，将稀疏的参数仅保存增量checkpoint，会极大的降低频繁保存checkpoint带来的额外开销。在PS挂掉后能尽量通过一个最近的全量的Checkpoint配合一系列的增量检查点恢复PS上最近一次训练完成的模型参数，减少重复计算。
## 用户接口
```python
def tf.train.MonitoredTrainingSession(..., save_incremental_checkpoint_secs=None, ...):pass
```

`tf.train.MonitoredTrainingSession`接口增加参数`save_incremental_checkpoint_secs`，默认值为`None`，用户可以设置以秒为单位的`incremental_save checkpoint`的时间，使用增量checkpoint功能

## 代码示例
使用高层API（`tf.train.MonitoredTrainingSession`）
```python
import tensorflow as tf
import time, os

tf.logging.set_verbosity(tf.logging.INFO)

sparse_var=tf.get_variable("a", shape=[30,4], initializer=tf.ones_initializer(tf.float32),partitioner=tf.fixed_size_partitioner(num_shards=4))
dense_var=tf.get_variable("b", shape=[30,4], initializer=tf.ones_initializer(tf.float32),partitioner=tf.fixed_size_partitioner(num_shards=4))

ids=tf.placeholder(dtype=tf.int64, name='ids')
emb = tf.nn.embedding_lookup(sparse_var, ids)

fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')

gs = tf.train.get_or_create_global_step()

opt=tf.train.AdagradOptimizer(0.1, initial_accumulator_value=1)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v, global_step=gs)

path = 'test/test4tolerance/ckpt/'
with tf.train.MonitoredTrainingSession(checkpoint_dir=path,
        save_checkpoint_secs=60,
        save_incremental_checkpoint_secs=20) as sess:
  for i in range(1000):
    print(sess.run([gs, train_op, loss], feed_dict={"ids:0": i%10}))
    time.sleep(1)
```

## 模型导出
在默认情况下，无法将增量checkpoint相关子图导出到SavedModel中，如果用户希望在Serving中通过“增量模型更新”来支持秒级更新，就需要将增量相关子图导出到SavedModel。目前需要使用DeepRec提供的[Estimator](https://github.com/AlibabaPAI/estimator)来导出。

示例代码：
```python
estimator.export_saved_model(
    export_dir_base,
    serving_input_receiver_fn,
    ... 
    save_incr_model=True)
```

需要注意：

当构图的时候没有增量模型图的时候配置 save_incr_model=True 会报错，所以图中只有全量，save_incr_model只能配置false(default值)。当图中有全量和增量，save_incr_model 配置true，SavedModel图可以加载全量或增量的模型。若将save_incr_model 配置false，SavedModel图只能加载全量的模型。

