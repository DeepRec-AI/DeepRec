# Embedding Variable GPU支持
## 功能介绍
GPU具有强大的并行计算能力，对于EmbeddingVariable底层的Hash Table查找、插入等操作也具有明显的加速作用。同时，对于模型计算部分若使用GPU，则使用GPU上的EmbeddingVariable也可避免Host和Device上的数据拷贝，提高整体性能。因此我们增加了EmbeddingVariable的GPU支持。

当前版本的EmbeddingVariable GPU实现暂时只支持部分基础功能。对于特征淘汰、特征准入、特征统计等功能暂未支持。对应的优化器现在提供了Adagrad以及FtrlOptimizer的支持。


## 使用方法
**配置环境变量**

下面的TF_CUDA_COMPUTE_CAPABILITIES（cuCollection需要7.5及以上）
```
export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0"
```

**编译**
configure时GPU_EV配置Yes，编译打开EmbeddingVaraible on GPU版本。

**使用**
使用开启了GPU支持的DeepRec版本，在拥有NVIDIA GPU的环境下，EmbeddingVariable会自动被放置在GPU device上。

我们也可手动指定device，将其放置于GPU上
```python
with tf.device('/gpu:0'):
    var = tf.get_embedding_variable("var_0",
                                    embedding_dim=3,
                                    initializer=tf.ones_initializer(tf.float32),
                                    partitioner=tf.fixed_size_partitioner(num_shards=4))
```

或者使用feature_column
```python
columns = tf.feature_column.categorical_column_with_embedding("col_emb", dtype=tf.dtypes.int64)
with tf.device('/gpu:0'):
    W = tf.feature_column.embedding_column(categorical_column=columns,
                dimension=3,
                initializer=tf.ones_initializer(tf.dtypes.float32))
```

注意：目前GPU EV不支持incremental checkpoint，如果使用的话EV相关的OP会被放置到CPU上，这个问题我们后续会修复。
