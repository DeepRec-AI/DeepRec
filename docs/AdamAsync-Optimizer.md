# AdamAsync Optimizer
## 介绍
大规模分布式异步训练时，原生Tensorflow的Adam Optimizer实现存在一些问题，例如分布式训练速度提升不上去，部分PS节点Load异常的高等问题。
​

针对Adam Optimizer在异步训练时遇到的问题，实现了AdamAsyncOptimizer：

1. 为每个variables创建伴生的beta1_power和beta2_power的slot，去掉全局依赖；
1. Optimizer在apply gradient到某个variable时，同时更新其伴生的beta1_power和beta2_power；
1. adam的计算公式修改为原始版本，解决NAN问题；
1. 修改后的计算公式如下：
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

5. 对于sparse variables，在apply gradient时，做momentum会对降低稀疏特征的更新幅度；
5. 在apply sparse variables时，我们提供一个开关（默认关闭），打开开关时可以将更新算法由adam换成rmsprop，去掉momentum的滑动平均功能，供不同用户需求使用。
5. AdamAsync Optimizer的使用方法和AdamOptimizer一样，并且多了一个可配置参数：`apply_saprse_rmpprop`，在apply sparse时是否启动rmsprop算法，默认是关闭的。
## 用户接口
训练时只需要定义`tf.train.AdamAsyncOptimizer`即可，和其他TF原生Optimizer使用方式相同。具体定义如下：
```python
class AdamAsyncOptimizer(optimizer.Optimizer):
def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
               use_locking=False, apply_sparse_rmsprop=False, name="AdamAsync"):

# 调用方法：
optimizer = tf.train.AdamAsyncOptimizer(
               learning_rate_new,
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

