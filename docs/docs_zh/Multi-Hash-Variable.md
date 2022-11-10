# Multi-Hash Variable
## 背景
在深度学习场景中，为了训练ID类的特征（例如user id或者item id），通常是将id类的稀疏特征映射到对应的低维稠密embedding向量中。在推荐场景中，随着数据量的逐渐增大，为每一个id类特征都存储一个对应的embedding向量，即无冲突Hash方案（EmbeddingVariable）会导致模型通常要比有冲突Hash方法下的模型大（Variable）。为了解决更大的模型带来的Training/Inference性能和资源的开销，DeepRec中已提供了若干功能来解决性能和资源问题，1）特征淘汰，通过将过期特征淘汰清理，避免过期特征积累过多，浪费内存；2）特征准入，低频特征准入机制，避免低频特征过拟合的同时，降低内存占用；3）混合多级EmbeddingVariable，通过将冷门特征存储在廉价更大的存储介质来支持更大的模型；4）DynamicDimensionEmbeddingVariable，通过使用策略让冷门特征使用更小的dim来表达，来降低冷门特征对存储空间的占用。本文中介绍的Multi-Hash Variable功能是另一种降低内存/显存占用的方法。
​

《Compositional Embeddings Using Complementary Partitions for Memory-Efficent Recommendation Systems》paper中提出了一种新的思路来降低内存占用的方法，具体方法是：

1. 构造若干个小的Embedding table
2. 每个对应的table对应一个hash function， 这些hash function需要互补，即对于每一个id，他对应的hash值集合是唯一的，与其他任何一个id都不完全相同。例如当有两个embedding table的时候，使用Quotient-Reminder可以保证每个key都有唯一的hash集合，具体如下图：

![img_1.png](Multi-Hash-Variable/img_1.png)

3. 根据一定的策略将从多个table里取出来的embedding组合成最终的emebdding，例如add、multiply以及concat。
## Multi-Hash Variable
为了在DeepRec中提供上述的功能，我们在DeepRec实现了Multi-Hash Variable功能。目前可以通过`get_multihash_variable`接口来使用该功能，接口如下：
```python
def get_multihash_variable(name,
                           dims,
                           num_of_partitions=2,
                           complementary_strategy="Q-R",
                           operation="add",
                           dtype=float,
                           initializer=None,
                           regularizer=None,
                           trainable=None,
                           collections=None,
                           caching_device=None,
                           partitioner=None,
                           validate_shape=True,
                           use_resource = None,
                           custom_getter=None,
                           constraint=None,
                           synchronization=VariableSynchronization.AUTO,
                           aggregation=VariableAggregation.NONE):

#name: multihash variable的名字
#embedding dim：需要传入一个list， 如果list的长度是1，那么operation必须从add或mult中选择；
                如果list长度大于1，那么operation必须选择concat，同时list中的元素加起来长度要等于embedding_dim
#num_of_partions: variable的partition的数量。如果complementary_strategy为“Q-R”, 那么该
                  参数必须为2.
#complementary_strategy: 目前支持“Q-R”
#operation：从"add", "mult","concat中三选一
#intialier：与variable的相同
#partitioner：与variable的相同
```
目前只支持两个partition使用QR方法的多哈希，原因是根据paper中的实验，这种方法已经可以支持大部分的场景，同时超过三个partition的方法相对来说比较复杂，还会带来更多的lookup开销。
​

**使用示例：**
使用`get_multihash_variable`接口
```python
import tensorflow as tf

def main(unused_argv):
  embedding = tf.get_multihash_variable("var-dist",
                                         [[2,2],[2,2]],
                                         complementary_strategy="Q-R",
                                         operation="concat",
                                         initializer=tf.ones_initializer)
  var = tf.nn.embedding_lookup(embedding, [0,1,2,3])
  fun = tf.multiply(var, 2.0, name='multiply')
  loss1 = tf.reduce_sum(fun, name='reduce_sum')
  opt = tf.train.AdagradOptimizer(0.1)
  g_v = opt.compute_gradients(loss1)
  train_op = opt.apply_gradients(g_v)
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
       sess.run([init])
       print(sess.run([var, train_op]))
       print(sess.run([var, train_op]))
       print(sess.run([var, train_op]))

if __name__=="__main__":
  tf.app.run()
```
使用`categorical_column_with_multihash`接口
```python
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.feature_column import feature_column_v2 as fc2

columns = fc2.categorical_column_with_multihash("col_emb", dims = (2,2))
W = tf.feature_column.embedding_column(categorical_column=columns,
            dimension=(2,3),
            initializer=tf.ones_initializer(tf.dtypes.float32))

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3]], values=tf.cast([0,1,2,3], tf.dtypes.int64), dense_shape=[4, 4])

emb = tf.feature_column.input_layer(ids, [W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss1 = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.AdagradOptimizer(0.1)
g_v = opt.compute_gradients(loss1)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op]))
    print(sess.run([emb, train_op]))
    print(sess.run([emb, train_op]))
```
