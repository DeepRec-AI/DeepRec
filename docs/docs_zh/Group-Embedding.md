# Group Embedding

## 背景
DeepRec的EmbeddingVariable之前针对tensorflow原生的embedding lookup API做了单路Embeddig子图Fusion（主要是将EmbeddingLookup中子图的多个Op做fusion），
但在典型的CTR场景，实际上会存在有个Embedding的查询，这种情况我们经过profiling发现在ps端仍然会存在并发查询带来的多个Op的kernel launch的问题，
因此我们希望能支持多个Embedding的同时聚合查询，从而优化该场景下Embedding查询的性能。

Group Embedding功能支持同时对多个EmbeddingVariable GPU聚合查询，Embedding可以被设置在不同的GPU上（默认是根据使用的GPU的数量均匀分布）。
接口支持单卡的Fusion以及分布式的多卡的Fusion。分布式的多卡Fusion实现的方式是将多路Embedding的keys通过集合通信分发给相应的GPU，查询完成后，
得到的Embedding values同样以集合通信的方式广播到每个worker上。单卡的Fusion则是通过一个大的FusionOp将多路Embedding的查询合并。下面我们分单机和分布式两种使用方法给大家介绍。

## 单机模式

### 用户接口
单机模式下Group Embedding的接口支持两个层面的API，分别为底层API `tf.nn.group_embedding_lookup_sparse` 
和基于feature_column的API `tf.feature_column.group_embedding_column_scope` 。

**group_embedding_lookup_sparse**

```python
def group_embedding_lookup_sparse(params,
                                  sp_ids,
                                  combiners,
                                  partition_strategy="mod",
                                  sp_weights=None,
                                  name=None):
```

- `params` : List, 该参数可以接收一个或者多个EmbeddingVariable或者是原生Tensorflow Variable
- `sp_ids` : List | Tuple , SparseTensor ，values是用于查找的ID 长度必须和params保持一致
- `combiners` : List | Tuple 查找完得到的embedding tensor聚合的方式，支持 `mean` 和 `sum`
- `partition_strategy` : str 目前暂时不支持
- `sp_weights` : List | Typle sp_ids 的 values 的权重。
- `name` : str group的名称

**group_embedding_column_scope**

```python
def group_embedding_column_scope(name=None):
```

- `name` ： scope的名称

我们只需要先初始化一个上下文 `group_embedding_column_scope` 并且在这个上下文内完成`EmbeddingColumn`类的构造，
在后续传入 `tf.feature_column.input_layer` 时会将这些EmbeddingColumn自动做聚合查询。接口的底层实现是基于`tf.SparseTensor`设计的。

### 使用示例

**group_embedding_lookup的使用方式如下**

```python
import tensorflow as tf

ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                    filter_option=None)

with tf.device('/GPU:{}'.format(0)):
    var_0 = tf.get_embedding_variable("var_0",
                                    embedding_dim=16,
                                    initializer=tf.ones_initializer(tf.float32),
                                    ev_option=ev_opt)
    
    var_1 = tf.get_embedding_variable("var_1",
                                    embedding_dim=8,
                                    initializer=tf.ones_initializer(tf.float32),
                                    ev_option=ev_opt)

##推荐使用SparseTensor的表示
indices_0 = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], 
                            values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64),
                            dense_shape=[5, 5])  

embedding_weights = [var_0, var_1]
indices = [indices_0 for _ in range(2)]
combiners = ["sum", "sum"]

deep_features = tf.nn.group_embedding_lookup_sparse(embedding_weights,
                                                    indices,
                                                    combiners)

init = tf.global_variables_initializer()
sess_config = tf.ConfigProto()
sess_config.gpu_options.visible_device_list = "0"
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
  sess.run(init)
  print("init global done")
  print(sess.run([deep_features]))
```

**feature_column的使用方式如下**

```python
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.framework import dtypes

ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                    filter_option=None)

# group_name 代表需要聚合的embedding的分组，默认为空（不做聚合）
with tf.device('/gpu:0'), tf.feature_column.group_embedding_column_scope(name="item")::
  ad0_col = tf.feature_column.categorical_column_with_embedding(
      key='ad0', dtype=dtypes.int64, ev_option=ev_opt)
  ad0_fc = tf.feature_column.embedding_column(
    categorical_column=ad0_col,
    dimension=20,
    initializer=tf.constant_initializer(0.5))
  ad1_fc = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_embedding(
      key='ad1', dtype=dtypes.int64, ev_option=ev_opt),
    dimension=30,
    initializer=tf.constant_initializer(0.5))

with tf.device('/gpu:0'), tf.feature_column.group_embedding_column_scope(name="user")::
  user0_fc = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_embedding(
        key='user0', dtype=dtypes.int64, ev_option=ev_opt),
      dimension=20,
      initializer=tf.constant_initializer(0.5))

columns = [ad0_fc, ad1_fc, user0_fc]

ids={}
ids["ad0"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])    
ids["ad1"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])
ids["user0"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])   

emb = tf.feature_column.input_layer(ids, columns)
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

sess_config = tf.ConfigProto()
sess_config.gpu_options.visible_device_list = "0" 
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([fun, train_op,loss]))
```


## 分布式模式

### 配置使用环境
首先我们需要保证编译好了SOK,SOK的编译步骤如下：
```bash
bazel --output_base /tmp build -j 16  -c opt --config=opt  //tensorflow/tools/pip_package:build_sok && ./bazel-bin/tensorflow/tools/pip_package/build_sok
```

### 用户接口
1. 使用分布式模式的Group Embedding的接口需要开启设置`tf.config.experimental.enable_distributed_strategy()`
参数配置: 
- ```strategy="collective"```,该模式推荐用于单机多卡的训练，会完成horovod模块以及SOK相关依赖的初始化。
- ```strategy="parameter_server"```,该模式推荐用于有PS角色的分布式训练，不依赖任何模块（功能实现中）

2. 分布式模式同样支持两个层面的API，分别为底层API `tf.nn.group_embedding_lookup_sparse` 和基于feature_column的API `tf.feature_column.group_embedding_column_scope` 。
  **提示**：单机和分布式模式在使用接口上另一个区别是分布式模式的sp_id适合RaggedTensor的输入，单机模式适合SparseTensor的输入。

### 使用示例

**group_embedding_lookup的使用方式如下**

```python
import tensorflow as tf
##分布式模式需要开启该配置
tf.config.experimental.enable_distributed_strategy(strategy="collective")

ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                    filter_option=None)

with tf.device('/GPU:{}'.format(0)):
    var_0 = tf.get_embedding_variable("var_0",
                                    embedding_dim=16,
                                    initializer=tf.ones_initializer(tf.float32),
                                    ev_option=ev_opt)
    
    var_1 = tf.get_embedding_variable("var_1",
                                    embedding_dim=8,
                                    initializer=tf.ones_initializer(tf.float32),
                                    ev_option=ev_opt)

##推荐使用RaggedTensor的表示
indices_0 = tf.RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2, 6],
    row_splits=[0, 4, 4, 7, 8, 8])

embedding_weights = [var_0, var_1]
indices = [indices_0 for _ in range(2)]
combiners = ["sum", "sum"]

deep_features = tf.nn.group_embedding_lookup_sparse(embedding_weights,
                                                    indices,
                                                    combiners)

init = tf.global_variables_initializer()
sess_config = tf.ConfigProto()
sess_config.gpu_options.visible_device_list = "0"
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
  sess.run(init)
  print("init global done")
  print(sess.run([deep_features]))
```

**feature_column的使用方式如下**

```python
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.framework import dtypes
##分布式模式需要开启该配置
tf.config.experimental.enable_distributed_strategy(strategy="collective")

ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                    filter_option=None)

with tf.device('/gpu:0'), tf.feature_column.group_embedding_column_scope(name="item")::
  ad0_col = tf.feature_column.categorical_column_with_embedding(
      key='ad0', dtype=dtypes.int64, ev_option=ev_opt)
  ad0_fc = tf.feature_column.embedding_column(
    categorical_column=ad0_col,
    dimension=20,
    initializer=tf.constant_initializer(0.5))
  ad1_fc = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_embedding(
      key='ad1', dtype=dtypes.int64, ev_option=ev_opt),
    dimension=30,
    initializer=tf.constant_initializer(0.5))

with tf.device('/gpu:0'), tf.feature_column.group_embedding_column_scope(name="user")::
  user0_fc = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_embedding(
        key='user0', dtype=dtypes.int64, ev_option=ev_opt),
      dimension=20,
      initializer=tf.constant_initializer(0.5))

columns = [ad0_fc, ad1_fc, user0_fc]

ids={}
ids["ad0"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])    
ids["ad1"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])
ids["user0"] = tf.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                        values=tf.cast([1,3,2,3,4,5,3], tf.dtypes.int64), dense_shape=[5, 5])   

emb = tf.feature_column.input_layer(ids, columns)
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
init = tf.global_variables_initializer()

sess_config = tf.ConfigProto()
sess_config.gpu_options.visible_device_list = "0"
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([fun, train_op,loss]))
```

详细的使用还可以参考modelzoo中提供的[DCNv2模型示例](../../modelzoo/features/group_embedding/dcnv2/train.py)

**Benchmarks**

我们用DCNv2模型测试了GroupEmbedding的性能， 训练的数据集是Critro，batch_size大小为512.
**collective**
| API | Global-step/s | Note |
| ------ | ---------------------- | ---- |
| tf.nn.group_embedding_lookup_sparse            | 85.3 (+/- 0.1)  | 1 card |
| tf.nn.group_embedding_lookup_sparse            | 122.2 (+/- 0.2) | 2 card |
| tf.nn.group_embedding_lookup_sparse            | 212.4 (+/- 0.4) | 4 card |
| tf.feature_column.group_embedding_column_scope | 79.7 (+/- 0.1)  | 1 card |
| tf.feature_column.group_embedding_column_scope | 152.2 (+/- 0.2) | 2 card |
| tf.feature_column.group_embedding_column_scope | 272 (+/- 0.4)   | 4 card |

**localized**
| tf.nn.group_embedding_lookup_sparse(Variable)  | 153.2 (+/- 0.1)  | 1 card |
