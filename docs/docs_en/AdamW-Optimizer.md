# AdamW Optimizer
## Introduction
The AdamW optimizer supports EmbeddingVariable, which adds the weight decay function compared to the Adam optimizer.

This is a kind of implementation of the AdamW optimizer which is mentioned in Loshch ilov & Hutter (https://arxiv.org/abs/1711.05101) "Decoupled Weight Decay Regularization".

## User interface
You need to use the `tf.train.AdamWOptimizer` function interface during training, which is the same as other TF native Optimizers. The specific definition is as follows:
```python
class AdamWOptimizer(DecoupledWeightDecayExtension, adam.AdamOptimizer):
  def __init__(self,
               weight_decay,
               learning_rate=0.001,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               use_locking=False,
               name="AdamW"):

# call function
optimizer = tf.train.AdamWOptimizer(
               weight_decay=weight_decay_new
               learning_rate=learning_rate_new,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8)
```
## Example
```python
import tensorflow as tf

var = tf.get_variable("var_0", shape=[10,16],
                       initializer=tf.ones_initializer(tf.float32))

emb = tf.nn.embedding_lookup(var, tf.cast([0,1,2,5,6,7], tf.int64))
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')

gs= tf.train.get_or_create_global_step()
opt = tf.train.AdamWOptimizer(weight_decay=0.01, learning_rate=0.1)

g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)

init = tf.global_variables_initializer()

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=sess_config) as sess:
  sess.run([init])
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))
```

