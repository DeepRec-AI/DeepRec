# EmbeddingVariable进阶功能：特征淘汰
## 功能介绍
对于一些对训练没有帮助的特征，我们需要将其淘汰以免影响训练效果，同时也能节约内存。在DeepRec中我们支持了特征淘汰功能，每次存ckpt的时候会触发特征淘汰，目前我们提供了两种特征淘汰的策略：

- 基于global step的特征淘汰功能：第一种方式是根据global step来判断一个特征是否要被淘汰。我们会给每一个特征分配一个时间戳，每次前向该特征被访问时就会用当前的global step更新其时间戳。在保存ckpt的时候判断当前的global step和时间戳之间的差距是否超过一个阈值，如果超过了则将这个特征淘汰（即删除）。这种方法的好处在于查询和更新的开销是比较小的，缺点是需要一个int64的数据来记录metadata，有额外的内存开销。 用户通过配置**steps_to_live**参数来配置淘汰的阈值大小。
- 基于l2 weight的特征淘汰： 在训练中如果一个特征的embedding值的L2范数越小，则代表这个特征在模型中的贡献越小，因此在存ckpt的时候淘汰淘汰L2范数小于某一阈值的特征。这种方法的好处在于不需要额外的metadata，缺点则是引入了额外的计算开销。用户通过配置**l2_weight_threshold**来配置淘汰的阈值大小。

## 使用方法
用户可以通过以下的方法使用特征淘汰功能

```python
#使用global step特征淘汰
evict_opt = tf.GlobalStepEvict(steps_to_live=4000)

#使用l2 weight特征淘汰：
evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)

ev_opt = tf.EmbeddingVariableOption(evict_option=evict_opt)

#通过get_embedding_variable接口使用
emb_var = tf.get_embedding_variable("var", embedding_dim = 16, ev_option=ev_opt)

#通过sparse_column_with_embedding接口使用
from tensorflow.contrib.layers.python.layers import feature_column
emb_var = feature_column.sparse_column_wth_embedding("var", ev_option=ev_opt)

emb_var = tf.feature_column.categorical_column_with_embedding("var", ev_option=ev_opt)
```
下面是特征淘汰接口的定义
```python
@tf_export(v1=["GlobalStepEvict"])
class GlobalStepEvict(object):
  def __init__(self,
               steps_to_live = None):
    self.steps_to_live = steps_to_live

@tf_export(v1=["L2WeightEvict"])
class L2WeightEvict(object):
  def __init__(self,
               l2_weight_threshold = -1.0):
    self.l2_weight_threshold = l2_weight_threshold
    if l2_weight_threshold <= 0 and l2_weight_threshold != -1.0:
      print("l2_weight_threshold is invalid, l2_weight-based eviction is disabled")
```
参数解释：

- `steps_to_live`：Global step特征淘汰的阈值，如果特征超过`steps_to_live`个global step没有被访问过，那么则淘汰
- `l2_weight_threshold`: L2 weight特征淘汰的阈值，如果特征的L2-norm小于阈值，则淘汰

功能开关：

如果没有配置`GlobalStepEvict`以及`L2WeightEvict`、`steps_to_live`设置为`None`以及`l2_weight_threshold`设置小于0则功能关闭，否则功能打开。