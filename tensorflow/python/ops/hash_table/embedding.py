# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-short-docstring-punctuation

"""
Embedding ops for hash table, with hook.
"""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.hash_table import hash_table, admit_strategy
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_hash_ops
# Imports gradient definitions.
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

class EmbeddingLookupContext(object):
  def __init__(
      self, input_hash_table, origin_key, partitioned_keys,
      filtered_keys, ids):
    self._hash_table = input_hash_table
    self._origin_key = origin_key
    self._partitioned_keys = partitioned_keys
    self._filtered_keys = filtered_keys
    self._ids = ids
    if isinstance(input_hash_table, hash_table.DistributedHashTable):
      self._partitions = input_hash_table.partitions
    else:
      self._partitions = [input_hash_table]

    self._device_dict = {}
    for i in range(len(self._partitions)):
      self._device_dict[self._partitions[i].device] = {
        "hash_table":self._partitions[i],
        "filtered_key":filtered_keys[i],
        "id":ids[i],
      }

    self._result = {}

  def hash_table(self):
    return self._hash_table

  def devices(self):
    return self._device_dict.keys()

  def origin_key(self):
    return self._origin_key

  def partitioned_keys(self):
    return self._partitioned_keys

  def partitions(self):
    return self._partitions

  def ids(self):
    return self._ids

  def origin_keys(self):
    return self._origin_key

  def filtered_keys(self):
    return self._filtered_keys

  def hash_table_on_device(self, device):
    return self._device_dict[device]["hash_table"]

  def filtered_key_on_device(self, device):
    return self._device_dict[device]["filtered_key"]

  def id_on_device(self, device):
    return self._device_dict[device]["id"]

  def result(self):
    return self._result

  def add_result(self, name, rst):
    if name in self._result:
      raise RuntimeError("Double set result with name: " + name)
    self._result[name] = rst

  def add_update_op(self, op):
    collection = ops.get_collection_ref(ops.GraphKeys.UPDATE_OPS)
    collection.append(op)

@tf_export("hash_table.EmbeddingLookupHook")
class EmbeddingLookupHook(object):
  def get_admit_strategy_factory(self, distributed_hash_table):
    if 'get_admit_strategy' in dir(self):
      return self.get_admit_strategy
    else:
      return None
  def on_embedding_lookup(self, ctx):
    return
  def __enter__(self):
    get_embedding_lookup_scope().add_hook(self)
  def __exit__(self, exc_type, exc_val, exc_tb):
    get_embedding_lookup_scope().remove_hook(self)

  @abc.abstractmethod
  def get_config(self):
    raise NotImplementedError('Embedding lookup hook get config function not implemented.')

@tf_export("hash_table.BloomFilterLookupHook")
class BloomFilterLookupHook(EmbeddingLookupHook):
  def __init__(self, minimum_frequency, max_element_size=None,
      false_positive_probability=None, name=None):
    super(BloomFilterLookupHook, self).__init__()
    self._minimum_frequency = minimum_frequency
    self._max_element_size = max_element_size
    self._false_positive_probability = false_positive_probability
    self._name = name

  def get_config(self):
    return {
      'minimum_frequency': self._minimum_frequency,
      'max_element_size': self._max_element_size,
      'false_positive_probability': self._false_positive_probability,
      'name': self._name
      }

  def get_admit_strategy_factory(self, distributed_hash_table):
    def wrapper(hash_table):
      return admit_strategy.BloomFilterAdmitStrategy(self._minimum_frequency,
          self._max_element_size, self._false_positive_probability,
          slicer=hash_table.slicer, hash_table=hash_table,
          distributed_name=hash_table.distributed_name + '_BloomFilter',
          name=self._name).handle
    return wrapper

  def on_embedding_lookup(self, ctx):
    return

class EmbeddingLookupScope(object):
  def __init__(self):
    self._hooks = []

  def add_hook(self, hook):
    self._hooks.append(hook)

  def remove_hook(self, hook):
    if self._hooks[-1] is not hook:
      raise ValueError("hooks should be a stack")
    self._hooks = self._hooks[:-1]

  def hooks(self):
    return list(self._hooks)

  def embedding_lookup_distribute_hash(self, hash_table, origin_keys, default_value=0, counts=None, name=None):
    with ops.name_scope(name, "EmbeddingLookupScope_EmbeddingLookup") as name:
      admit_strategy_factory = None
      for hook in self._hooks:
        admit_strategy_factory_x = hook.get_admit_strategy_factory(hash_table)
        if admit_strategy_factory_x is not None:
          if admit_strategy_factory is not None:
            raise RuntimeError("Only one embedding hook should set admit strategy")
          admit_strategy_factory = admit_strategy_factory_x
      keys, gather_ids, indices, dense_shape = hash_table.gen_ids(origin_keys,
          admit_strategy_factory, counts)
      result = hash_table.lookup_by_id(gather_ids, indices, dense_shape, default_value)
      filtered_keys = [None] * len(keys)
      if admit_strategy_factory is not None:
        masks = [math_ops.not_equal(ids, -1) for ids in gather_ids]
        filtered_keys = [array_ops.boolean_mask(keys[i], masks[i]) for i in range(len(keys))]
      ctx = EmbeddingLookupContext(hash_table, origin_keys, keys, filtered_keys, gather_ids)
      for hook in self._hooks:
        hook.on_embedding_lookup(ctx)
      return result, ctx.result()

  def embedding_lookup_hash(self, hash_table, origin_keys, default_value=0, counts=None, name=None):
    with ops.name_scope(name, "EmbeddingLookupScope_EmbeddingLookup") as name:
      admit_strategy_factory = None
      for hook in self._hooks:
        admit_strategy_factory_x = hook.get_admit_strategy_factory(hash_table)
        if admit_strategy_factory_x is not None:
          if admit_strategy_factory is not None:
            raise RuntimeError("Only one embedding hook should set admit strategy")
          admit_strategy_factory = admit_strategy_factory_x
      if admit_strategy_factory is None:
        admit_strategy = lambda ht: None
      else:
        admit_strategy = admit_strategy_factory
      ids = hash_table.gen_ids(origin_keys,
          admit_strategy(hash_table), counts)
      result = hash_table.lookup_by_id(ids, default_value)
      filtered_key = None
      if admit_strategy_factory is not None:
        mask = math_ops.not_equal(ids, -1)
        filtered_key = array_ops.boolean_mask(origin_keys, mask)
      ctx = EmbeddingLookupContext(hash_table, origin_keys, [origin_keys], [filtered_key], [ids])
      for hook in self._hooks:
        hook.on_embedding_lookup(ctx)
      return result, ctx.result()

_EMBEDDINGSCOPE_KEY = "embedding_lookup_scope_key"

def get_embedding_lookup_scope():
  """Returns the current embedding scope."""
  scope = ops.get_collection(_EMBEDDINGSCOPE_KEY)
  if scope:
    return scope[0]
  scope = EmbeddingLookupScope()
  ops.add_to_collection(_EMBEDDINGSCOPE_KEY, scope)
  return scope

@tf_export("hash_table.embedding_lookup")
def embedding_lookup(input_hash_table, origin_keys, default_value=0, counts=None, name=None):
  if isinstance(input_hash_table, hash_table.DistributedHashTable):
    lookup_func = get_embedding_lookup_scope().embedding_lookup_distribute_hash
  elif isinstance(input_hash_table, hash_table.HashTable):
    lookup_func = get_embedding_lookup_scope().embedding_lookup_hash
  else:
    raise NotImplementedError(
      "Hash table embedding lookup only support"
      " `DistributedHashTable` or `HashTable`")
  return lookup_func(input_hash_table, origin_keys, default_value, counts, name)

#TODO: add a api to return embedding result 2
@tf_export("hash_table.embedding_lookup_sparse")
def embedding_lookup_sparse(hash_table,
                            sp_ids,
                            sp_weights,
                            default_value=0,
                            name=None,
                            combiner=None,
                            sp_uniq_ids=None):
  if combiner is None:
    logging.warn("The default value of combiner will change from \"mean\" "
                 "to \"sqrtn\" after 2016/11/01.")
    combiner = "mean"
  if combiner not in ("mean", "sqrtn", "sum", "tile"):
    raise ValueError("combiner must be one of 'mean', 'sqrtn', 'tile' or 'sum'")
  if not isinstance(sp_ids, sparse_tensor.SparseTensor):
    raise TypeError("sp_ids must be SparseTensor")
  ignore_weights = sp_weights is None
  if not ignore_weights:
    if not isinstance(sp_weights, sparse_tensor.SparseTensor):
      raise TypeError("sp_weights must be either None or SparseTensor")
    sp_ids.values.get_shape().assert_is_compatible_with(
        sp_weights.values.get_shape())
    sp_ids.indices.get_shape().assert_is_compatible_with(
        sp_weights.indices.get_shape())
    sp_ids.dense_shape.get_shape().assert_is_compatible_with(
        sp_weights.dense_shape.get_shape())
    # TODO(yleon): Add enhanced node assertions to verify that sp_ids and
    # sp_weights have equal indices and shapes.

  with ops.name_scope(name, "embedding_lookup_sparse") as name:
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != dtypes.int32:
      segment_ids = math_ops.cast(segment_ids, dtypes.int32)

    ids = sp_ids.values
    counts = None
    if ignore_weights:
      if sp_uniq_ids is not None:
        ids, idx = sp_uniq_ids
      else:
        ids, idx, counts = array_ops.unique_with_counts(ids)
    else:
      idx = None

    embeddings, rst = embedding_lookup(
        hash_table, ids, default_value, counts, name)
    if not ignore_weights:
      weights = sp_weights.values
      if weights.dtype != embeddings.dtype:
        weights = math_ops.cast(weights, embeddings.dtype)

      # Reshape weights to allow broadcast
      ones = array_ops.fill(
          array_ops.expand_dims(array_ops.rank(embeddings) - 1, 0), 1)
      bcast_weights_shape = array_ops.concat([array_ops.shape(weights), ones],
                                             0)

      orig_weights_shape = weights.get_shape()
      weights = array_ops.reshape(weights, bcast_weights_shape)

      # Set the weight shape, since after reshaping to bcast_weights_shape,
      # the shape becomes None.
      if embeddings.get_shape().ndims is not None:
        weights.set_shape(
            orig_weights_shape.concatenate(
                [1 for _ in range(embeddings.get_shape().ndims - 1)]))

      embeddings *= weights

      if combiner == "sum":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
      elif combiner == "mean":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weight_sum = math_ops.segment_sum(weights, segment_ids)
        embeddings = math_ops.div(embeddings, weight_sum)
      elif combiner == "sqrtn":
        embeddings = math_ops.segment_sum(embeddings, segment_ids)
        weights_squared = math_ops.pow(weights, 2)
        weight_sum = math_ops.segment_sum(weights_squared, segment_ids)
        weight_sum_sqrt = math_ops.sqrt(weight_sum)
        embeddings = math_ops.div(embeddings, weight_sum_sqrt)
      elif combiner == "tile":
        column_ids = sp_ids.indices[:, 1]
        embeddings = _tile_combine_embedding(embeddings,
                                             segment_ids,
                                             column_ids,
                                             sp_ids.dense_shape)
      else:
        assert False, "Unrecognized combiner"
    else:
      assert idx is not None
      if combiner == "sum":
        embeddings = math_ops.sparse_segment_sum(
            embeddings, idx, segment_ids)
      elif combiner == "mean":
        embeddings = math_ops.sparse_segment_mean(
            embeddings, idx, segment_ids)
      elif combiner == "sqrtn":
        embeddings = math_ops.sparse_segment_sqrt_n(
            embeddings, idx, segment_ids)
      elif combiner == "tile":
        embeddings = array_ops.gather(embeddings, idx)
        column_ids = sp_ids.indices[:, 1]
        embeddings = _tile_combine_embedding(embeddings,
                                             segment_ids,
                                             column_ids,
                                             sp_ids.dense_shape)
      else:
        assert False, "Unrecognized combiner"

    return embeddings, rst

def _tile_combine_embedding(embeddings, segment_ids, column_ids, sp_shape):
  column_ids = math_ops.cast(column_ids, dtypes.int32)
  sp_shape = math_ops.cast(sp_shape, dtypes.int32)
  segment_ids = segment_ids * sp_shape[1] + column_ids
  total_size = sp_shape[0] * sp_shape[1]
  embeddings = math_ops.unsorted_segment_sum(embeddings, segment_ids, total_size)
  embeddings = array_ops.reshape(
      embeddings, [sp_shape[0], sp_shape[1] * array_ops.shape(embeddings)[-1]])
  return embeddings

@tf_export("hash_table.ReadOnlyHook")
class ReadOnlyHook(EmbeddingLookupHook):
  def __init__(self):
    self.admit = {}

  def get_admit_strategy(self, ht):
    if ht.device not in self.admit:
      with ops.device(ht.device):
        self.admit[ht.device] = gen_hash_ops.read_only_hash_table_admit_strategy_op()
    return self.admit[ht.device]

  def get_config(self):
    return {}
