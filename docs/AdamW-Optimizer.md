# AdamW Optimizer
## 介绍
Optimizer that implements the Adam algorithm with weight decay.
​

This is an implementation of the AdamW optimizer described in "Decoupled Weight Decay Regularization" by Loshch ilov & Hutter ([https://arxiv.org/abs/1711.05101](https://arxiv.org/pdf/1711.05101.pdf)).

It computes the update step of `tf.train.Adam` and additionally decays the variable. Note that this is different from adding L2 regularization on the variables to the loss: it regularizes variables with large gradients more than L2 regularization would, which was shown to yield better training loss and generalization error in the paper above.


## 用户接口
训练时只需要定义`tf.train.AdamWOptimizer`即可，和其他TF原生Optimizer使用方式相同。具体定义如下：
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

# 调用方法：
optimizer = tf.train.AdamWOptimizer(
               weight_decay=weight_decay_new
               learning_rate=learning_rate_new,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8)
```
## 使用示例
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

