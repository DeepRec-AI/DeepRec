# Group Embedding

## 背景
DeepRec的EmbeddingVariable之前针对tensorflow原生的embedding lookup API做了单路Embeddig子图Fusion（主要是将EmbeddingLookup中子图的多个Op做fusion），
但在典型的CTR场景，实际上会存在有个Embedding的查询，这种情况我们经过profiling发现仍然会存在在ps端并发查询带来的多个Op的kernel launch的问题，
在这种情况下我们希望能完善多个Embedding的同时聚合查询，从而优化该场景的Embedding查询性能

Group Embedding功能支持同时对多个EmbeddingVariable GPU聚合查询，Embedding可以被设置在不同的GPU上（默认是根据使用的GPU的数量均匀分布）。
接口支持单卡的Fusion以及多卡的Fusion。目前只提供多卡的Fusion，所有的keys会通过集合通信分发给相应的GPU，查询完成后，得到的Embedding values同样以集合通信的方式广播到每个worker上。

## 使用方法

### 配置使用环境
首先我们需要保证编译好了SOK,SOK的编译和验证步骤如下：
```bash
bazel --output_base /tmp build -j 16  -c opt --config=opt  //tensorflow/tools/pip_package:build_sok && ./bazel-bin/tensorflow/tools/pip_package/build_sok
```

### 用户接口
1. 首先，使用Group Embedding的接口需要开启设置`tf.config.experimental.enable_group_embedding()`
参数配置: 
- ```fusion_type="collective"```,该模式推荐用于单机多卡的训练，会完成horovod模块以及SOK相关依赖的初始化。
- ```fusion_type="localized"```,该模式推荐用于单机单卡的训练，不依赖任何模块
  **提示**：上述两个模式在使用接口上唯一的区别是`collective`的sp_id适合RaggedTensor的输入，`localized`适合SparseTensor的输入

2. 我们支持两个层面的API，分别为底层API `tf.nn.group_embedding_lookup_sparse` 和基于feature_column的API `tf.feature_column.group_embedding_column_scope` 。

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
- `sp_weights` : List | Typle sp_ids 的 values 的权重。(目前暂时不支持，后续会开放)
- `name` : str group的名称

**group_embedding_column_scope**

```python
def group_embedding_column_scope(name=None):
```
- `name` ： scope的名称

我们只需要先初始化一个上下文`group_embedding_column_scope`
并且在这个上下文内完成`EmbeddingColumn`类的构造，在后续传入 `tf.feature_column.input_layer` 时会将这些EmbeddingColumn自动做聚合查询。值得注意的是接口的底层实现是基于`tf.RaggedTensor`设计的。查询Embedding的IDS虽然同时也支持`SparseTensor`但最后仍然会转换为`RaggedTensor`的表示，这会引入一定的性能开销。

### 使用示例

**group_embedding_lookup的使用方式如下**

```python
import tensorflow as tf

tf.config.experimental.enable_group_embedding(fusion_type="collective")

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

deep_features = tf.nn.group_embedding_lookup_sparse(embedding_weights, indices, combiners)

init = tf.global_variables_initializer()
sess_config = tf.ConfigProto()
sess_config.gpu_options.visible_device_list = "0" #str(hvd.local_rank())
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
tf.config.experimental.enable_group_embedding(fusion_type="collective")

ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                    filter_option=None)

# group_name 代表需要聚合的embedding的分组，默认为空（不做聚合）
with tf.device('/gpu:0'), tf.feature_column.group_embedding_column_scope(name="item")::
  ad0_col = tf.feature_column.categorical_column_with_embedding(
      key='ad0', dtype=dtypes.int64, ev_option=ev_opt)
  ad0_fc = tf.feature_column.embedding_column(
    categorical_column=ad0_col,
    dimension=20,
    initializer=tf.constant_initializer(0.5),
    group_name='item')
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
sess_config.gpu_options.visible_device_list = "0" #(hvd.local_rank())
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    sess.run(init)
    print("init global done")
    print(sess.run([fun, train_op,loss]))
```

详细的使用还可以参考modelzoo中提供的[DCNv2模型示例](../../modelzoo/features/group_embedding_lookup/dcnv2/train.py)

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
