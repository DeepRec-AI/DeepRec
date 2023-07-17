# Collective Training

## 背景
对于像DLRM类似的稀疏的推荐模型，通常有大量的参数以及复杂的矩阵运算。PS的异步训练范式，难以充分利用集群中的GPU来加速整个训练/推理过程。我们开始尝试将所有的参数放置在worker上，但是大规模参数(Embedding)占据了大量的显存导致无法储存在单个GPU上，我们需要对参数做并行切分存放。原生Tensorflow不支持模型并行(MP)，社区已有的许多优秀的基于Tensorflow实现的addons，比如HybridBackend（以下简称HB）、SparseOperationKit(以下简称SOK）等等。DeepRec没有重复开发，而是提供了一个统一的同步训练的接口`CollectiveStrategy`供用户自行选择使用。用户可以以很少的代码改动来使用不同的同步训练框架。

## 接口介绍

1. 目前接口支持HB和SOK，用户可以通过环境变量 `COLLECTIVE_STRATEGY`来选择。`COLLECTIVE_STRATEGY`可以配置hb，sok分别对应HB和SOK方式。与正常启动Tensorflow任务的区别在于，用户使用同步训练的时候需要通过额外的模块拉起，需要通过以下方式启动:

```bash
CUDA_VISIBLE_DEVICES=0,1 COLLECTIVE_STRATEGY=hb python3 -m tensorflow.python.distribute.launch <python script.py>
```
如果环境变量没有配置`CUDA_VISIBLE_DEVICES`，进程会默认拉起当前环境GPU数目的训练子进程。

2. 用户脚本中则需要初始化一个`CollectiveStrategy`,来完成模型的构建。
```python
class CollectiveStrategy:
    def scope(self, *args, **kwargs):
        pass
    def embedding_scope(self, **kwargs):
        pass
    def world_size(self):
        pass
    def rank(self):
        pass
    def estimator(self):
        pass
    def export_saved_model(self):
        pass
```

使用同步该接口有以下几个步骤
- 在整个模型定义前用strategy.scope()标记
- 在需要模型并行的地方(embedding层)使用embedding_scope()标记
- 在导出的时候使用export_saved_model
- (Optional)strategy还提供estimator接口给用户使用。

## 使用示例

**MonitoredTrainingSession**

下面例子指导用户如何通过tf.train.MonitoredTrainingSession构图
```python
import tensorflow as tf
from tensorflow.python.distribute.group_embedding_collective_strategy import CollectiveStrategy

#STEP1:  initialize a collective strategy
strategy = CollectiveStrategy()
#STEP2:  define the data parallel scope
with strategy.scope(), tf.Graph().as_default():
    #STEP3:  define the model parallel scope
    with strategy.embedding_scope():
        var = tf.get_variable(
            'var_1',
            shape=(1000, 3),
            initializer=tf.ones_initializer(tf.float32),
            partitioner=tf.fixed_size_partitioner(num_shards=strategy.world_size())
        )
    emb = tf.nn.embedding_lookup(
        var, tf.cast([0, 1, 2, 5, 6, 7], tf.int64))
    fun = tf.multiply(emb, 2.0, name='multiply')
    loss = tf.reduce_sum(fun, name='reduce_sum')
    opt = tf.train.FtrlOptimizer(
        0.1,
        l1_regularization_strength=2.0,
        l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    with tf.train.MonitoredTrainingSession('') as sess:
        emb_result, loss_result, _ = sess.run([emb, loss, train_op])
        print (emb_result, loss_result)
```

**Estimator**

下面例子指导用户如何通过tf.estimator.Estimator构图
```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.distribute.group_embedding_collective_strategy import CollectiveStrategy

#STEP1:  initialize a collective strategy
strategy = CollectiveStrategy()
#STEP2:  define the data parallel scope
with strategy.scope(), tf.Graph().as_default():
    def input_fn():
        ratings = tfds.load("movie_lens/100k-ratings", split="train")
        ratings = ratings.map(
            lambda x: {
                "movie_id": tf.strings.to_number(x["movie_id"], tf.int64),
                "user_id": tf.strings.to_number(x["user_id"], tf.int64),
                "user_rating": x["user_rating"]
            })
        shuffled = ratings.shuffle(1_000_000,
                                    seed=2021,
                                    reshuffle_each_iteration=False)
        dataset = shuffled.batch(256)
        return dataset

    def input_receiver():
        r'''Prediction input receiver.
        '''
        inputs = {
            "movie_id": tf.placeholder(dtype=tf.int64, shape=[None]),
            "user_id": tf.placeholder(dtype=tf.int64, shape=[None]),
            "user_rating": tf.placeholder(dtype=tf.float32, shape=[None])
        }
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def model_fn(features, labels, mode, params):
        r'''Model function for estimator.
        '''
        del params
        movie_id = features["movie_id"]
        user_id = features["user_id"]
        rating = features["user_rating"]

        embedding_columns = [
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_embedding(
                    "movie_id", dtype=tf.int64),
                dimension=16,
                initializer=tf.random_uniform_initializer(-1e-3, 1e-3)),
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_embedding(
                    "user_id", dtype=tf.int64),
                dimension=16,
                initializer=tf.random_uniform_initializer(-1e-3, 1e-3))
            ]
        #STEP3:  define the model parallel scope
        with strategy.embedding_scope():
            with tf.variable_scope(
                'embedding',
                partitioner=tf.fixed_size_partitioner(
                strategy.world_size)):
                deep_features = [
                    tf.feature_column.input_layer(features, [c])
                    for c in embedding_columns]
        emb = tf.concat(deep_features, axis=-1)
        logits = tf.multiply(emb, 2.0, name='multiply')

        if mode == tf.estimator.ModeKeys.TRAIN:
            labels = tf.reshape(tf.to_float(labels), shape=[-1, 1])
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))
            step = tf.train.get_or_create_global_step()
            opt = tf.train.AdagradOptimizer(learning_rate=self._args.lr)
            train_op = opt.minimize(loss, global_step=step)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_chief_hooks=[])

        return None
    estimator = strategy.estimator(model_fn=model_fn,
                                   model_dir="./",
                                   config=None)
    estimator.train_and_evaluate(
      tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=50),
      tf.estimator.EvalSpec(
        input_fn=input_fn))
    estimator.export_saved_model("./", input_receiver)
```

## 附录

- 目前DeepRec提供了相应的GPU镜像给用户使用(alideeprec/deeprec-release:deeprec2304-gpu-py38-cu116-ubuntu20.04-hybridbackend),用户也可以参考Dockerfile(../../cibuild/dockerfiles/Dockerfile.devel-py3.8-cu116-ubuntu20.04-hybridbackend)
- Modelzoo中关于上述两个使用方法还提供了更详细的demo，参见[ModelZoo](../../modelzoo/features/grouped_embedding)

- 如果需要做进一步的优化，关于HB和SOK自身还有更多的微调参数，可以参考:
[SOK](./SOK.md) 和 [HB](https://github.com/DeepRec-AI/HybridBackend)