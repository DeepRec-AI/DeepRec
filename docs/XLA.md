# XLA

XLA（加速线性代数）是一种针对特定领域的线性代数编译器，能够加快 TensorFlow 模型的运行速度，而且可能完全不需要更改源代码。使用方式和原生Tensorflow完全一致。

## 全局开启xla
```python
sess_config = tf.ConfigProto()
...
sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

...

with tf.train.MonitoredTrainingSession(
    ...
    config=sess_config) as sess:
    ...

```

## 控制开启的scope
```python
def dnn(self, dnn_input, dnn_hidden_units=None, layer_name=''):
    # 增加下面代码可以限定部分graph使用xla来编译
    #
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
    with jit_scope():
       for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with tf.variable_scope(layer_name + "_%d" % layer_id,
                                   partitioner=self.dense_layer_partitioner,
                                   reuse=tf.AUTO_REUSE) as dnn_layer_scope:
                dnn_input = tf.layers.dense(dnn_input,
                                            units=num_hidden_units,
                                            activation=tf.nn.relu,
                                            name=dnn_layer_scope)
                if self.use_bn:
                    dnn_input = tf.layers.batch_normalization(
                        dnn_input, training=self._is_training, trainable=True)
                add_layer_summary(dnn_input, dnn_layer_scope.name)

       return dnn_input
```
