# Feature Eviction of EmbeddingVariable

## Introduction

For some features that are not helpful for training, we need to eliminate them so as not to affect the training effect, and also save memory. In DeepRec, we support the feature eviction, which triggers feature elimination every time checkpoint is saved. Currently, we provide two strategies for feature eviction:

- Feature eviction based on global step: The first method is to judge whether a feature should be eliminated according to the global step. We will assign a timestamp to each feature, which will be updated with the current global step each time the feature is updated in backward. When saving ckpt, it is judged whether the gap between the current global step and the timestamp exceeds a threshold, and if so, this feature is eliminated (that is, deleted). The advantage of this method is that the query and update overhead is relatively small, but the disadvantage is that an int64 data is required to record metadata, which has additional memory overhead. The user configures the threshold size of elimination by configuring the **steps_to_live** parameter.
- Feature eviction based on l2 weight: In training, if the L2 norm of the embedding value of a feature is smaller, it means that the contribution of this feature in the model is smaller. Therefore, when ckpt is saved, the L2 norm is smaller than a certain threshold. Characteristics. The advantage of this method is that no additional metadata is required, but the disadvantage is that it introduces additional computational overhead. The user configures the threshold size for elimination by configuring **l2_weight_threshold**.

## API

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

Description:

- `steps_to_live`: The threshold for global step feature eviction, if the feature has not been visited for more than `steps_to_live` global steps, then it will be eliminated
- `l2_weight_threshold`: The threshold for L2 weight feature elimination, if the L2-norm of the feature is less than the threshold, it will be eliminated

## Usage

If `GlobalStepEvict` or `L2WeightEvict` is not configured, `steps_to_live` is set to `None` and `l2_weight_threshold` is set to less than 0, then the feature is disabled, otherwise the feature is enabled.

Users can use the feature eviction through the following methods.

```python
# global step feature eviction
evict_opt = tf.GlobalStepEvict(steps_to_live=4000)

# l2 weight feature eviction
evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)

ev_opt = tf.EmbeddingVariableOption(evict_option=evict_opt)

# get_embedding_variable interface
emb_var = tf.get_embedding_variable("var", embedding_dim = 16, ev_option=ev_opt)

# sparse_column_with_embedding interface
from tensorflow.contrib.layers.python.layers import feature_column
emb_var = feature_column.sparse_column_with_embedding("var", ev_option=ev_opt)

emb_var = tf.feature_column.categorical_column_with_embedding("var", ev_option=ev_opt)
```
