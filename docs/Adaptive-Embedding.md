# Adaptive Embedding
## 背景
在无冲突的Embedding中，如果稀疏特征非常多，会导致EmbeddingVariable（以下简称EV）占用非常大的空间，一方面在训练的是后对PS内存造成较大的压力，另一方面模型线上服务的时候会删减模型，通常的手段为通过一个静态filter过滤特征，这种过滤规则过于生硬，没有很好的考虑动态学习中的Embedding，不可避免的会对最终学习的结果产生影响。同时，由于特征的出现的频度不同，而EV由于无冲突对于所有的特征都一律从initializer设定的初始值（一般设为0）开始学起，那么对于出现不怎么多的特征，会学的很慢；同时对于一些出现频次从低到高的特征，也需要逐渐学习到一个较好的状态，不能共享别的特征的学习结果。这也是EV本身的完全无冲突的思想造成的。
## 功能说明
使用静态Variable和动态EV共同存储稀疏模型，对于新加入的特征存于可冲突的静态Variable，对于出现频率较高的特征存于无冲突的EV，Embedding迁移到EV可以复用在静态Variable的学习结果，极大的降低模型大小。
## 用户接口
adaptive embedding功能需要用户通过feature_column API使用
```python
def categorical_column_with_adaptive_embedding(key,
                                               hash_bucket_size,
                                               dtype=dtypes.string,
                                               partition_num=None,
                                               ev_option=variables.EmbeddingVariableOption())

# key, feature_column 名称
# hash_bucket_size, 静态Variable第一维度大小
# dtype, 稀疏特征类型, 默认为string
# partition_num, EV partition数量, 默认不分片
# ev_option, EV配置信息
```
## 代码示例
```python
import tensorflow as tf

columns = tf.feature_column.categorical_column_with_adaptive_embedding("col_emb", hash_bucket_size=100, dtype=tf.int64)
W = tf.feature_column.embedding_column(categorical_column=columns,
                                       dimension=3,
                                       initializer=tf.ones_initializer(tf.float32))
ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0]],
                                 values=tf.cast([1,2,3,4,5,6,7,8,9,0], tf.int64),
                                 dense_shape=[10, 1])
adaptive_mask_tensors={}
adaptive_mask_tensors["col_emb"] = tf.cast([1,0,1,0,1,0,0,1,0,1], tf.int32)
emb = tf.feature_column.input_layer(ids, [W], adaptive_mask_tensors=adaptive_mask_tensors)

fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')

graph=tf.get_default_graph()

opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run([init])
  emb1, top, l = sess.run([emb, train_op, loss])
  emb1, top, l = sess.run([emb, train_op, loss])
  emb1, top, l = sess.run([emb, train_op, loss])
  emb1, top, l = sess.run([emb, train_op, loss])
  print(emb1, top, l)
```
