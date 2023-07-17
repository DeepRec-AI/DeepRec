# Collective Training

## Background

For sparse recommendation models like DLRM, there are a large number of parameters and heavy GEMM operations. The asynchronous training paradigm of PS makes it difficult to fully utilize the GPUs in the cluster to accelerate the entire training/inference process.We try to place all the parameters on the worker, but the large amount of memory consumed by the parameters(Embedding) cannot be stored on a single GPU, so we need to perform sharding to place on all GPUs.Native Tensorflow did not support model parallel training (MP), and the community has many excellent plug-ins based on Tensorflow, such as HybridBackend (hereinafter referred to as HB), SparseOperationKit (hereinafter referred to as SOK), and so on. DeepRec provides a unified synchronous training interface `CollectiveStrategy` for users to choose and use. Users can use different synchronous training frameworks with very little code.

## Interface Introduction

1. Currently the interface supports HB and SOK, users can choose through the environment variable `COLLECTIVE_STRATEGY`. `COLLECTIVE_STRATEGY` can configure hb, sok corresponding to HB and SOK respectively. The difference from normal startup of Tensorflow tasks is that when users use synchronous training, they need to pull up through additional modules, which need to be started in the following way:

```bash
CUDA_VISIBLE_DEVICES=0,1 COLLECTIVE_STRATEGY=hb python3 -m tensorflow.python.distribute.launch <python script.py>
```
If the environment variable is not configured with `CUDA_VISIBLE_DEVICES`, the process will pull up the training sub-processes with the number of GPUs in the current environment by default.

2. In the user script, a `CollectiveStrategy` needs to be initialized to complete the construction of the model.

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

Following steps below to using synchronous training:
- Mark with strategy.scope() before the entire model definition.
- Use the embedding_scope() flag where model parallelism is required (embedding layer)
- Use export_saved_model when exporting
- (Optional) In addition, the strategy also provides the estimator interface for users to use.

## Example

**MonitoredTrainingSession**

The following example guides users how to construct Graph through tf.train.MonitoredTrainingSession.

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

The following example guides users how to construct Graph through tf.estimator.Estimator.
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

## Appendix

- Currently DeepRec provides the corresponding GPU image for users to use (alideeprec/deeprec-release:deeprec2304-gpu-py38-cu116-ubuntu20.04-hybridbackend), users can also refer to [Dockerfile](../../cibuild/dockerfiles/Dockerfile.devel-py3.8-cu116-ubuntu20.04-hybridbackend)
- We also provides more detailed demos about the above two usage methods, see: [ModelZoo](../../modelzoo/features/grouped_embedding)

- If further optimization is required, there are more fine-tuning parameters for HB and SOK, please refer to:
[SOK](./SOK.md) 和 [HB](https://github.com/DeepRec-AI/HybridBackend)
