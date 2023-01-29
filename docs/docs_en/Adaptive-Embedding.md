# Adaptive Embedding
## Introduction
If there are many sparse features, it will cause EmbeddingVariable (abbreviate as EV) to occupy lots of memory. On the one hand, it will put a lot of pressure on PS memory during training. On the other hand, when the model is served online, the model needs to be shrunk. The usual method is to filter features through a static filter. However, this filtering rule is too rigid, and the embedding in dynamic learning is not well considered, which will inevitably affect the final accuracy. Moreover, due to the different frequencies of features, and EV starts to learn all features from the initial value set by the initializer (generally set to 0), it will learn very slowly for features that do not appear very much. For some features with low to high frequency of occurrence, it is also necessary to gradually learn to a better state, and the learning results of other features cannot be shared. This is caused by the conflict-free thinking of EV.

Adaptive Embedding uses a static Variable and a dynamic EV to store sparse features together. For low-frequency features, it is stored in conflictable static Variable, and for features with high frequency, it is stored in non-conflicting EV. Migrating embedding to EV can reuse the learning results in the static Variable, which greatly reduces the model size.
## API
Adaptive Embedding needs to be used through the feature_column API
```python
def categorical_column_with_adaptive_embedding(key,
                                               hash_bucket_size,
                                               dtype=dtypes.string,
                                               partition_num=None,
                                               ev_option=variables.EmbeddingVariableOption())

# key, name of feature_column
# hash_bucket_size, the first dimension of static Variable
# dtype, data type of sparse features
# partition_num, partition number of EV, no partition by default
# ev_option, configuration of EV
```
## Example
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
