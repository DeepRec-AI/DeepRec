# Embedding Variable
## Introduction

Embedding parameters are saved in the form of tf.Variable in TensorFlow, and the size of Variable is [vocabulary_size, embedding_dimension] which needed to be decided before training and inference. This will bring difficulties for users in large-scale scenarios:

1. vocabulary_size is decided by the number of features, which makes it difficult to estimate vocabulary_size in online learning because of new features that keep coming.
2. The data type of the ids is string and their size are very large generally, so users need to hash the ids into values between 0 to vocabulary_size before lookup the embeddings:
    - The probability of different ids being mapped to the same embedding will increase significantly when the vocabulary_size is too small.;
    - Lots of memory will be wasted when the vocabulary_size is too big;
3. The enormous size of Embedding tables is the main reason for the increase of the model size. Even if the embedding of some features has little effect on the model through regularization, these embeddings cannot be removed from the model.

To solve the problems mentioned above, DeepRec provides EmbeddingVariable to support Variable with dynamic shape. With EmbeddingVariable, users can use memory efficiently while don't affect the accuracy of models, and making it easier to deploy the large-scale models.

EmbeddingVariable has undergone several iterations and currently support feature filter, feature eviction, feature statistics and other fundamental features. Moreover, optimizations including optimization on structure of sparse feature, lockless hash map, multi-tier embedding storage, Embedding GPU Ops, and GPU HashTable are added to EmbeddingVariable. Some features of EmbeddingVariable are also supported in TensorFlow recommenders-addons (https://github.com/tensorflow/recommenders-addons/pull/16).

## User APIs
DeepRec provides users with three ways to create EmbeddingVariable:

**Create EmbeddingVariable with `get_embedding_variable` API**
```python
def get_embedding_variable(name,
                           embedding_dim,
                           key_dtype=dtypes.int64,
                           value_dtype=None,
                           initializer=None,
                           regularizer=None,
                           trainable=True,
                           collections=None,
                           caching_device=None,
                           partitioner=None,
                           validate_shape=True,
                           custom_getter=None,
                           constraint=None,
                           init_data_source=None,
                           ev_option = tf.EmbeddingVariableOption()):
```

- `name`: name of EmbeddingVariable
- `embedding_dim`: dim of each embedding, e.g. 8, 64
- `key_dtype`: data type of the key used to lookup embedding, default is int64, allowed values are int64 and int32
- `value_dtype`: the data type of embedding parameters, currently limited to float 
- `initializer`: initial value of embedding parameters, initializer and list can be passed in
- `trainable`: whether to be added to the GraphKeys.TRAINABLE_VARIABLES collection
- `partitioner`: partition function
- `ev_option`: options of EmbeddingVariabe, e.g. options of feature filter and options of multi-tier storage

**Create EmbeddingVariable with `tf.feature_colum` API**
```python
def categorical_column_with_embedding(key,
                                  dtype=dtypes.string,
                                  partition_num=None,
                                  ev_option=tf.EmbeddingVariableOption()
                                  )
```

**Create EmbeddingVariable with `tf.contrib.feature_column` API**
```python
def sparse_column_with_embedding(column_name,
                                 dtype=dtypes.string,
                                 partition_num=None,
                                 steps_to_live=None,
                                 init_data_source=None,
                                 ht_partition_num=1000,
                                 evconfig = variables.EmbeddingVariableOption()
```

## Demo

**With `get_embedding_variable` API**
```python
import tensorflow as tf

var = tf.get_embedding_variable("var_0",
                                embedding_dim=3,
                                initializer=tf.ones_initializer(tf.float32),
                                partitioner=tf.fixed_size_partitioner(num_shards=4))

shape = [var1.total_count() for var1 in var]

emb = tf.nn.embedding_lookup(var, tf.cast([0,1,2,5,6,7], tf.int64))
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.AdagradOptimizer(0.1)

g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)

init = tf.global_variables_initializer()

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=sess_config) as sess:
  sess.run([init])
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))
  print(sess.run([shape]))
```
**With `categorical_column_with_embedding` API**
```python
import tensorflow as tf
from tensorflow.python.framework import ops


columns = tf.feature_column.categorical_column_with_embedding("col_emb", dtype=tf.dtypes.int64)
W = tf.feature_column.embedding_column(categorical_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=tf.cast([1,2,3,4,5], tf.dtypes.int64), dense_shape=[5, 5])

emb = tf.feature_column.input_layer(ids, [W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```
**With `sparse_column_with_embedding` API**
```python
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column


columns = feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=tf.dtypes.int64)
W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=tf.cast([1,2,3,4,5], tf.dtypes.int64), dense_shape=[5, 5])

emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=[W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```
**With `sequence_categorical_column_with_embedding` API**
```python
import tensorflow as tf
from tensorflow.python.feature_column import sequence_feature_column


columns = sequence_feature_column.sequence_categorical_column_with_embedding(key="col_emb", dtype=tf.dtypes.int32)
W = tf.feature_column.embedding_column(categorical_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,4]], \
                                 values=tf.cast([1,3,2,3,4,5], tf.dtypes.int64), 
                                 dense_shape=[5, 5])

emb, length = tf.contrib.feature_column.sequence_input_layer(ids, [W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```
**With `weighted_categorical_column` API**
```python
import tensorflow as tf


categorical_column = tf.feature_column.categorical_column_with_embedding("col_emb", dtype=tf.dtypes.int64)

ids={}
ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])    
ids['weight'] = [[2.0],[5.0],[4.0],[8.0],[3.0],[1.0],[2.5]]

columns = tf.feature_column.weighted_categorical_column(categorical_column, 'weight')

W = tf.feature_column.embedding_column(categorical_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))
emb = tf.feature_column.input_layer(ids, [W])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
    print(sess.run([emb, train_op,loss]))
```

## EV Initializer

We found that the accuracy will decrease significantly if glorot uniform initializer is not used when training WDL, DIN and DIEN (e.g. AUC increases very slowly and the best AUC decreases significantly when using ones_initializer). However, EmbeddingVariable requires a dynamic shape while glorot uniform initializer requires a static shape. Here are the methods to solve the problems caused by static shape in other frameworks:
  - PAI-TF140 & 1120: Generate a default value tensor with fixed shape in each step, which will affect the performance when users enlarge the size of default value tensor for better accuracy of models.
  - XDL: XDL first generates a default value matrix with static shape at initialization stage. A default value is fetched every time a new feature comes. A new default value matrix is generated when all default values are fetched. The advantage of this approach is it can promise every feature can have its unique default value. But on the other hand, this approach needs to set a mutex lock when generating the default value matrix, which will decrease performance. Moreover, the default value matrix generator in a temporarily constructed context doesn't have the info of the graph, so users can not fix the default value by setting the random seed.
  - Abacus: Abacus calls the initializer individually for each feature to generate a default value, which hardly affects the performance, but there are two disadvantages of this method. First, because the initializer only generates one default value every time, default values may not conform to the distribution. Second, if users set the random seed, default values of features will be the same.

Considering the above methods, we implement EV Initializer. EV Initializer generates a default value matrix at the initialization stage. Then every new feature fetches its default value according to the index calculated by its id mod a fixed size. EV Initializer first avoids the effect of mutex lock. Second, default values will conform to distribution. Finally users can fix the default value by setting the random seed.

### Usage
**Users can refer to the following examples to set EV Initializer**

```python
init_opt = tf.InitializerOption(initializer=tf.glorot_uniform_initializer,
                                default_value_dim = 10000)
ev_opt = tf.EmbeddingVariableOption(init_option=init)

#Create EmbeddingVariable with get_embedding_variable
emb_var = tf.get_embedding_variable("var", embedding_dim = 16, ev_option=ev_opt)

#Create EmbeddingVariable with sparse_column_wth_embedding
from tensorflow.contrib.layers.python.layers import feature_column
emb_var = feature_column.sparse_column_wth_embedding("var", ev_option=ev_opt)

#Create EmbeddingVariable with categorical_column_with_embedding
emb_var = tf.feature_column.categorical_column_with_embedding("var", ev_option=ev_opt)
```
**Here is the definition of InitializerOption**
```python
@tf_export(v1=["InitializerOption"])
class InitializerOption(object):
  def __init__(self,
               initializer = None,
               default_value_dim = 4096,
               default_value_no_permission = .0):
    self.initializer = initializer
    self.default_value_dim  = default_value_dim
    self.default_value_no_permission = default_value_no_permission
    if default_value_dim <=0:
      print("default value dim must larger than 1, the default value dim is set to default 4096.")
      default_value_dim = 4096
```

- `initializer`: the initializer of EmbeddingVariable, default is truncated normal initializer.
- `default_value_dim`: the number of default values generated by EV Initializer, the configuration can be referred to the hash bucket size or the number of features, the default value is 4096.
- `default_value_no_permission`: the  default value for filtered features when enabling feature filter.



