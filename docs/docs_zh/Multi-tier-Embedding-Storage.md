# Embedding多级存储
## 1. 功能介绍

在推荐系统中，稀疏参数在模型中通常有两种存储形式，一是大小动态增减的Embedding表（用于特征避免冲突的场景，例如：PAI-TF的EmbeddingVariable功能），二是大小固定的静态Embedding表（用于训练前就确定好稀疏参数的个数或者将个数不定的稀疏参数hash到某一固定大小，例如：以TF中固定大小的Variable当做Embedding表）。随着稀疏类型的深度学习模型规模逐渐变大，Embedding表需要更大的内存才能满足模型训练需求。

对于动态Embedding表，在模型训练之初很难估计整体模型的最终资源消耗（EmbeddingVariable的动态特性，即特征的动态伸缩，对应到系统资源，就是内存的动态申请与释放，这会在图编译期难以确定内存资源消耗），随着特征数量的增长，导致分布式训练PS节点经常遇到内存资源不足问题。在这种情况下，用户往往只能增加PS数目或者增大PS内存量来重新启动任务避免，此方案并没有从根本解决问题，只是通过申请更多的资源来应对当前的内存资源瓶颈，当模型的参数规模（特征量）超过一定数值，又会遇到同样的问题。

为了从根本上解决内存瓶颈的问题，我们在DeepRec中提供了基于EmbeddingVariable的Embedding多级存储功能。Embedding多级存储利用推荐场景中存在的特征访问倾斜的特性，结合cache策略，将热特征放在高速存储中，冷特征放在低速存储中，因此可以在适当降低性能的情况下大幅增加单机内可保存的参数量。通过充分利用单机内的存储资源可以解决存储资源不足的问题，有效地减少训练推理的成本和能耗。同时，Embedding多级存储保证训练或推理过程中embedding数据的使用和更新都发生在第一级存储中，因此模型效果和只使用单级存储时保持一致。

## 2. 用户接口
用户可以通过如下方式使用EmbeddingVariable多级存储功能
```python
import tensorflow as tf
from tensorflow.core.framework.embedding import config_pb2

storage_option = tf.StorageOption(storage_type=config_pb2.StorageType.DRAM_SSDHASH,
                                  storage_path="/tmp/ssd_utpy",
                                  storage_size=[512])
                                  
ev_opt = tf.EmbeddingVariableOption(storage_option=storage_option)
                                  
#通过get_embedding_variable接口使用
emb_var = tf.get_embedding_variable("var", embedding_dim = 16, ev_option=ev_opt)

#通过sparse_column_with_embedding接口使用
from tensorflow.contrib.layers.python.layers import feature_column
emb_var = feature_column.sparse_column_wth_embedding("var", ev_option=ev_opt)
                                  
#通过categorical_column_with_embedding接口使用
emb_var = tf.feature_column.categorical_column_with_embedding("var", ev_option=ev_opt)

```
下面是EmbeddingVariable多级存储的接口定义：
```python
@tf_export(v1=["StorageOption"])
class StorageOption(object):
  def __init__(self,
               storage_type=None,
               storage_path=None,
               storage_size=[1024*1024*1024]):
    self.storage_type = storage_type
    self.storage_path = storage_path
    self.storage_size = storage_size
```
参数解释：

- stroage_type：使用的存储类型， 例如DRAM_SSD为使用DRAM和SSD作为embedding的存储，具体支持的存储类型会在第4节中给出
- storage_path:   如果使用SSD存储，则需要配置该参数指定保存embedding数据的文件夹路径
- storage_size： 指定每个层级可以使用的存储容量，单位是字节，例如对于DRAM+PMem要使用1GB DRAM和 10GB PMem，则配置为[1024*1024*1024, 10*1024*1024*1024]，默认是每级1GB，目前的实现中无法限制SSD的使用量
## 3.使用示例
使用**get_embedding_variable**接口
```python
import tensorflow as tf
from tensorflow.core.framework.embedding import config_pb2
with tf.device('/cpu:0'): #目前只有CPU的EV算子支持使用多级存储
  storage_option = tf.StorageOption(storage_type=config_pb2.StorageType.DRAM_SSDHASH,
                                  storage_path="/tmp/ssd_utpy",
                                  storage_size=[128])
  ev_opt = tf.EmbeddingVariableOption(storage_option=storage_option)

  var = tf.get_embedding_variable("var_0",
                                embedding_dim=3,
                                initializer=tf.ones_initializer(tf.float32),
                                ev_option=ev_opt)

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
```
使用**categorical_column_with_embedding**接口的例子

```python
import tensorflow as tf
from tensorflow.core.framework.embedding import config_pb2

with tf.device('/cpu:0'): #目前只有CPU的EV算子支持使用多级存储
  storage_option = tf.StorageOption(storage_type=config_pb2.StorageType.DRAM_SSDHASH,
                                  storage_path="/tmp/ssd_utpy",
                                  storage_size=[128])
  ev_opt = tf.EmbeddingVariableOption(storage_option=storage_option)

  columns = tf.feature_column.categorical_column_with_embedding("col_emb", dtype=tf.dtypes.int64, ev_option=ev_opt)
  W = tf.feature_column.embedding_column(categorical_column=columns,
            dimension=3,
            initializer=tf.ones_initializer(tf.dtypes.float32))

  ids={}
  ids["col_emb"] = tf.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=tf.cast([1,2,3,4,5], tf.dtypes.int64), dense_shape=[5, 4])

  emb = tf.feature_column.input_layer(ids, [W])
  fun = tf.multiply(emb, 2.0, name='multiply')
  loss = tf.reduce_sum(fun, name='reduce_sum')
  opt = tf.train.AdagradOptimizer(0.1)
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
## 4. 支持的存储类型
计划支持的存储类型包括：

- HBM
- DRAM （已支持）
- HBM_DRAM
- HBM_DRAM_PMEM
- HBM_DRAM_LEVELDB
- HBM_DRAM_SSDHASH 
- HBM_DRAM_PMEM_LEVELDB 
- HBM_DRAM_PMEM_SSDHASH
- DRAM_PMEM （已支持）
- DRAM_LEVELDB（已支持）
- DRAM_SSDHASH （已支持）
- DRAM_PMEM_LEVELDB 
- DRAM_PMEM_SSDHASH

以下是各种存储介质的说明：

- HBM：GPU显存
- DRAM：CPU内存
- PMEM：持久化内存
- LevelDB：基于LevelDB开发的SSD存储
- SSDHASH：基于Hash索引的SSD存储，相比LevelDB实现，有更好的性能和内存稳定性。SSDHASH支持同步和异步两种compaction的方式。使用同步compaction时，向SSD写入数据和compaction将会使用同一个线程，异步时则各使用一个线程。
用户可以通过配置环境变量`TF_SSDHASH_ASYNC_COMPACTION`选择使用哪种compaction方式，当TF_SSDHASH_ASYNC_COMPACTION=1时打开异步compaction功能；设置为0或不设置时使用同步compaction。

## 5.设置淘汰线程数量

为了减少使用多级存储带来的性能开销并且维持系统存储占用量稳定，多级存储会启动后台线程来异步地将数据写入到下级存储中。考虑到在一些场景中(例如在线serving场景)CPU资源紧张，因此多级存储中使用一个统一的线程池来管理系统中所有使用多级存储的EV，用户可以根据实际情况通过配置`TF_MULTI_TIER_EV_EVICTION_THREADS`环境变量来设置线程池中的线程数。
