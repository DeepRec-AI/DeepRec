# Incremental Checkpoint

## Introduction

In large-scale sparse training, the data is skewed, and most of the data in adjacent full checkpoints remains unchanged. In this context, saving only incremental checkpoints for sparse parameters will greatly reduce the overhead caused by frequently saving checkpoints. After the PS failover, try to restore the model parameters of the latest training on the PS through a recent full checkpoint and a series of incremental checkpoints to reduce repeated calculations.

## API

```python
def tf.train.MonitoredTrainingSession(..., save_incremental_checkpoint_secs=None, ...):
  pass
```
extra parameters:

`save_incremental_checkpoint_secs`, default: `None`.
User can set the incremental_save checkpoint time in seconds, to generate the incremental checkpoint.

## Example
High-level API（`tf.train.MonitoredTrainingSession`）
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

Estimator

Configure parameters when constructing `EstimatorSpec`

`tf.train.Saver` and `tf.train.Scaffold` set `incremental_save_restore=True`，`tf.train.CheckpointSaverHook` set save incremental checkpoint interval `incremental_save_secs`

```
def model_fn(self, features, labels, mode, params):

  ...

  scaffold = tf.train.Scaffold(
               saver=tf.train.Saver(
                 sharded=True,
                 incremental_save_restore=True),
               incremental_save_restore=True)

  ...

  return tf.estimator.EstimatorSpec(
           mode,
           loss=loss,
           train_op=train_op,
           training_hooks=[logging_hook],
           training_chief_hooks=[
             tf.train.CheckpointSaverHook(
               checkpoint_dir=params['model_dir'],
               save_secs=params['save_checkpoints_secs'],
               save_steps=params['save_checkpoints_steps'],
               scaffold=scaffold,
               incremental_save_secs=120)])
```

## Model Export

By default, incremental checkpoint subgraphs cannot be exported to SavedModel. If users want to support second-level updates through "incremental model update" in Serving, they need to export incremental checkpoint subgraphs to SavedModel. You need to use the [Estimator](https://github.com/AlibabaPAI/estimator) provided by DeepRec to export incremental checkpoint subgraphs.

Example：
```python
estimator.export_saved_model(
    export_dir_base,
    serving_input_receiver_fn,
    ...
    save_incr_model=True)
```

Attention:

When there is no incremental model when building graph, an error will be reported when configuring save_incr_model=True, so there is only the full amount in the graph, and save_incr_model can only be configured with false (default value). When there are full and incremental models in the graph, save_incr_model is set to true, and the SavedModel graph can load full or incremental models. If save_incr_model is set to false, the SavedModel graph can only load the full model.
