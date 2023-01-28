# AdamAsync Optimizer
## Introduction
In the process of large-scale distributed asynchronous training, there are some problems in the implementation of Adam Optimizer of native Tensorflow, such as the speed of distributed training cannot be improved, and the load value of some PS nodes is very high.

To solve the problems encountered by Adam Optimizer during asynchronous training, we implemented AdamAsyncOptimizer:

1. Create associated beta1_power and beta2_power slots for each variable, thereby removing global dependencies;
2. When the Optimizer applies the gradient to a variable, it updates its associated beta1_power and beta2_power at the same time;
3. The calculation formula of adam is modified to the original version to solve the NAN problem;
4. The revised calculation formula is as follows:
```python
auto alpha = lr() * Eigen::numext::sqrt(T(1) - beta2_power(0)) /
                 (T(1) - beta1_power(0));

    // beta1 == μ
    // beta2 == ν
    // v     == n
    // var   == θ
    m.device(d) = m * beta1() + grad * (T(1) - beta1());
    v.device(d) = v * beta2() + grad.square() * (T(1) - beta2());
    if (use_nesterov) {
      var.device(d) -= ((grad * (T(1) - beta1()) + beta1() * m) * alpha) /
                       (v.sqrt() + epsilon());
    } else {
      var.device(d) -= (m * alpha) / (v.sqrt() + epsilon());
    }

    // update beta1_power && beta2_power
    beta1_power.device(d) = beta1_power * beta1();
    beta2_power.device(d) = beta2_power * beta2();
```

5. For sparse variables, when applying gradient, doing momentum will reduce the update rate of sparse features;
6. When applying sparse variables, we provide a bool variable (default false). When set to true, the update algorithm can be changed from adam to rmsprop, so that the sliding average function of momentum can be removed.
7. AdamAsync Optimizer is used in the same way as AdamOptimizer, and there is an additional configurable parameter: `apply_saprse_rmpprop`, whether to enable the rmsprop algorithm when apply sparse is disabled by default.

## User interface
You need to use the `tf.train.AdamAsyncOptimizer` interface during training, which is the same as other TF native Optimizers. The specific definition is as follows:
```python
class AdamAsyncOptimizer(optimizer.Optimizer):
def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, apply_sparse_rmsprop=False, name="AdamAsync"):

# call function
optimizer = tf.train.AdamAsyncOptimizer(
               learning_rate_new,
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
opt = tf.train.AdamAsyncOptimizer(0.1)

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

