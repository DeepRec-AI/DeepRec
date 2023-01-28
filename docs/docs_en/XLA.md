# XLA

XLA (Accelerated Linear Algebra) is a domain-specific linear algebra compiler that speeds up the running of TensorFlow models, possibly with no source code changes at all. The usage is the same as that of native Tensorflow.

## Enable XLA for full graph
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

## Enable XLA for subgraph
```python
def dnn(self, dnn_input, dnn_hidden_units=None, layer_name=''):
    # Add the following code to limit part of the graph to use XLA to compile.
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
