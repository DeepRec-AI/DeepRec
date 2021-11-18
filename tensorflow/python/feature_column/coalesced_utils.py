# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Defines functions common to coalesced feature column files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import collections
import copy
import contextlib

from tensorflow.core.framework import variable_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.util import nest

class CoalescedScopeBase(object):
  def __init__(self, name=None):
    self._columns = dict()
    self._coalesced_map = dict()
    self._output_map = dict()
    self._built = False
    if name is None:
      name = 'CoalescedScopeBase'
    self._name = name
    self._name_set = set()

  @property
  def name(self):
    return self._name

  def get_name(self):
    scope = variable_scope.get_variable_scope()
    index = 0
    while True:
      name = self._name if index == 0 else '{}_{}'.format(self._name, index)
      real_name = (scope.name + '/' + name) if scope.name else name
      if real_name not in self._name_set:
        self._name_set.add(real_name)
        return name
      index += 1

  def allowed_column_types(self):
    raise NotImplementedError("must be implemented in descendants")

  def add_column(self, column):
    if not isinstance(column, self.allowed_column_types()):
      raise ValueError('{} is not allowd for coalescing, must be {}'.format(
                       column, self.allowd_column_types()))
    if column.name in self._columns:
      raise ValueError('column {} already exists: {}'.format(column.name, column))
    self._columns[column.name] = column

  def build(self):
    raise NotImplementedError("must be implemented in descendants")

  def get_coalesced_column_by_column(self, column):
    if not self._built:
      raise RuntimeError('CoalescedScope not built yet, can only use result '
                         'outside of scope definition')
    name = column.name
    if name not in self._columns or name not in self._coalesced_map:
      raise ValueError('column {} not coalesced in any scope'.format(name))
    return self._coalesced_map[name]

  def get_dense_tensor_by_column(
      self, column, inputs, weight_collections=None, trainable=None):
    name = column.name
    if name in self._output_map:
      return self._output_map[name]

    coalesced_column = self.get_coalesced_column_by_column(column)
    embeddings = coalesced_column._get_dense_tensor(
        inputs, weight_collections, trainable)
    for column, embedding in zip(coalesced_column.columns, embeddings):
      self._output_map[column.name] = embedding
    return self._output_map[name]

  def get_dense_tensor_by_column_v2(
      self, column, transformation_cache, state_manager):
    name = column.name
    if name in self._output_map:
      return self._output_map[name]

    coalesced_column = self.get_coalesced_column_by_column(column)
    embeddings = coalesced_column.get_dense_tensor(
        transformation_cache, state_manager)
    for column, embedding in zip(coalesced_column.columns, embeddings):
      self._output_map[column.name] = embedding
    return self._output_map[name]

  def create_state_by_column(
      self, column):
    coalesced_column = self.get_coalesced_column_by_column(column)
    coalesced_column.get_or_create_embedding_weights()

  def get_coalesced_name_by_column(self, column):
    coalesced_column = self.get_coalesced_column_by_column(column)
    return coalesced_column.name


class EmbeddingAttributes(object):
  def __init__(self,
               dimension,
               dtype,
               initializer,
               combiner,
               trainable,
               hash_combiner='',
               bucket_size=None):
    self._dimension = dimension
    self._dtype = dtype
    self._initializer = initializer
    self._combiner = combiner
    self._trainable = trainable
    self._hash_combiner = hash_combiner
    self._bucket_size = bucket_size

  @property
  def dimension(self):
    return self._dimension

  @property
  def dtype(self):
    return self._dtype

  @property
  def initializer(self):
    return self._initializer

  @property
  def combiner(self):
    return self._combiner

  @property
  def trainable(self):
    return self._trainable

  @property
  def hash_combiner(self):
    return self._hash_combiner

  @property
  def bucket_size(self):
    return self._bucket_size

class CoalescedSaveSliceInfo(object):
  def __init__(self,
               full_name,
               full_shape,
               var_offset,
               var_shape,
               var_full_name,
               save_slices,
               tensor_slices):
    self._full_name = full_name
    self._full_shape = full_shape
    self._var_offset = var_offset
    self._var_shape = var_shape
    self._var_full_name = var_full_name
    self._save_slices = save_slices
    self._tensor_slices = tensor_slices

  @property
  def full_name(self):
    return self._full_name

  @property
  def full_shape(self):
    return self._full_shape

  @property
  def var_offset(self):
    return self._var_offset

  @property
  def var_shape(self):
    return self._var_shape

  @property
  def var_full_name(self):
    return self._var_full_name

  @property
  def save_slices(self):
    return self._save_slices

  @property
  def tensor_slices(self):
    return self._tensor_slices

  def slot_save_slice_info(self, slot_name):
    full_name = self._full_name + "/" + slot_name
    var_full_name = self._var_full_name + "/" + slot_name
    result = CoalescedSaveSliceInfo(full_name,
                                    self.full_shape,
                                    self.var_offset,
                                    self.var_shape,
                                    var_full_name,
                                    copy.deepcopy(self._save_slices),
                                    copy.deepcopy(self._tensor_slices))
    for info in result.save_slices:
      info.full_name += "/" + slot_name
      info.var_full_name += "/" + slot_name
    return result

  def to_proto(self):
    """Returns a SaveSliceInfoDef() proto.
    Args:
      export_scope: Optional `string`. Name scope to remove

    Returns:
      A `SaveSliceInfoDef` protocol buffer, or None if the `Variable` is not
    """
    return variable_pb2.SaveSliceInfoDef()

_embedding_signatures = collections.defaultdict(dict)

def get_embedding_signature():
  global _embedding_signatures
  return _embedding_signatures

def add_embedding_signature(column, dimension, combiner, initializer,
                            trainable, bucket_size, dtype=dtypes.float32,
                            hash_combiner=''):
  global _embedding_signatures
  if column in _embedding_signatures:
    raise ValueError('EmbeddingColumn already exists: {}'.format(column))
  _embedding_signatures[column] = EmbeddingAttributes(
      dimension, dtype, initializer, combiner, trainable, hash_combiner,
      bucket_size)

def make_cluster_signature(column, hashtable_column=False):
  if hashtable_column:
    attr = column
  else:
    global _embedding_signatures
    if column not in _embedding_signatures:
      raise ValueError('signautre not found for column: {}'.format(column))
    attr = _embedding_signatures[column]
  signature = {
      'dimension': str(tensor_shape.TensorShape(attr.dimension)),
      'dtype': dtypes.as_dtype(attr.dtype).name,
      'initializer': type(attr.initializer).__name__,
      'initializer_config': attr.initializer.get_config(),
  }
  return json.dumps(signature, sort_keys=True)

def _make_runtime_signature(column, hashtable_column=False):
  if hashtable_column:
    attr = column
  else:
    global _embedding_signatures
    if column not in _embedding_signatures:
      raise ValueError('signautre not found for column: {}'.format(column))
    attr = _embedding_signatures[column]
  signature = {
      'combiner': attr.combiner,
      'trainable': attr.trainable,
  }
  if hashtable_column:
    signature['filter_hook'] = [type(hook).__name__ for hook in attr.embedding_lookup_hooks]
    signature['filter_hook_config'] = [hook.get_config() for hook in attr.embedding_lookup_hooks]
  else:
    signature['hash_combiner'] = attr.hash_combiner
  return json.dumps(signature, sort_keys=True)

def get_signature_attributes(column):
  if column not in _embedding_signatures:
    raise ValueError('signautre not found for column: {}'.format(column))
  return _embedding_signatures[column]

def check_coalesced_columns_compatible(columns, hashtable_column=False):
  base = None
  for i, c in enumerate(columns):
    if base is None:
      base = make_cluster_signature(c, hashtable_column)
    elif make_cluster_signature(c, hashtable_column) != base:
      raise ValueError('signature of column 0 not match with column %d' % i)

def deduplicate_shared_embedding(columns):
  index = 0
  unique_columns = []
  indices_map = dict()
  for column in columns:
    if hasattr(column, 'embedding_name'):
      name = column.embedding_name
    else:
      name = column.name
    if name not in indices_map:
      unique_columns.append(column)
      indices_map[name] = index
      index += 1
    else:
      indices_map[name] = indices_map[name]
  return unique_columns, indices_map

def build_slice_info(columns, partitioner):
  global _embedding_signatures
  bucket_size_sum = 0

  # calculate slice length for each column
  parts_list = []
  start_index = 0
  save_slice_infos = []
  for c in columns:
    attr = _embedding_signatures[c]
    bucket_size = attr.bucket_size
    dimension = attr.dimension
    bucket_size_sum += bucket_size
    dtype = attr.dtype
    size = partitioner(shape=tensor_shape.as_shape(bucket_size), dtype=dtype)[0]
    step = bucket_size // size
    extra = bucket_size % size
    parts = [0] * size
    for i in range(size):
      parts[(start_index + i) % size] = step + 1 if i < extra else step
    parts_list.append(parts)
    start_index = (extra + start_index) % size

    offset = 0
    full_shape = [bucket_size, dimension]
    slice_list = []
    for i in range(len(parts)):
      var_offset = [offset, 0]
      var_shape = [parts[i], dimension]
      slice_list.append(variables.Variable.SaveSliceInfo(
          '', full_shape, var_offset, var_shape, var_full_name=''))
      offset += parts[i]
    save_slice_infos.append(slice_list)

  # check all columns have same number of partitions
  size = None
  for i, infos in enumerate(save_slice_infos):
    if i == 0:
      size = len(infos)
    elif size != len(infos):
      raise ValueError(
          'Coalesced columns should be partitioned to same size,'
          'but column 0 and column {} not equal: {} vs {}'.format(
              i, size, len(infos)))

  # calculate tensor slices
  tensor_slices = []
  for i in range(size):
    offset = 0
    tensor_slice_list = []
    for j in range(len(parts_list)):
      begin = offset
      offset += parts_list[j][i]
      tensor_slice_list.append((slice(begin, offset), slice(None)))
    tensor_slices.append(tensor_slice_list)

  return save_slice_infos, tensor_slices, bucket_size_sum

def _merge_sparse_tensor(tensors, tensor_rank=2):
  if not all(isinstance(t, SparseTensor) for t in tensors):
    raise ValueError("Expected inputs of SparseTensor")
  values = array_ops.concat([t.values for t in tensors], axis=0)
  row_counts = [t.dense_shape[0] for t in tensors]
  dense_shape = [math_ops.reduce_sum(row_counts)]
  for i in range(1, tensor_rank):
    column_counts = [t.dense_shape[i] for t in tensors]
    dense_shape.append(math_ops.reduce_max(column_counts))
  row_offset = array_ops.split(
      math_ops.cumsum(row_counts, exclusive=True), len(tensors))
  indices = []
  for i, t in enumerate(tensors):
    offset = array_ops.concat(
        [row_offset[i],
        math_ops.to_int64(array_ops.fill([tensor_rank - 1], 0))], axis=0)
    indices.append(t.indices + offset)
  result = SparseTensor(indices=array_ops.concat(indices, axis=0),
                        values=values,
                        dense_shape=dense_shape)
  return result, row_counts

def _safe_merge_sparse_tensor(tensor_pairs, format_rank=2):
  if not tensor_pairs:
    return tensor_pairs
  origin_shape_list = []
  format_id_tensors = []
  weight_values = []
  for id_tensor, weight_tensor in tensor_pairs:
    original_shape = id_tensor.dense_shape
    original_rank_dim = id_tensor.dense_shape.get_shape()[0]
    original_rank = (
        array_ops.size(original_shape)
        if original_rank_dim.value is None
        else original_rank_dim.value)
    diff_rank = original_rank - format_rank
    id_tensor = sparse_ops.sparse_reshape(
        id_tensor,
        array_ops.concat([[
          math_ops.reduce_prod(
            array_ops.slice(original_shape, [0], [diff_rank + 1]))],
        original_shape[-format_rank + 1:]], 0))
    format_id_tensors.append(id_tensor)
    if weight_tensor is not None:
      weight_values.append(weight_tensor.values)
    else:
      weight_values.append(None)
    origin_shape_list.append(math_ops.cast(original_shape, dtypes.int32))
  merged_ids, row_counts = _merge_sparse_tensor(format_id_tensors, tensor_rank=format_rank)

  if all([w is None for w in weight_values]):
    merged_weights = None
  else:
    weights_values = array_ops.concat(weight_values,
                                      axis=0)
    merged_weights = SparseTensor(indices=merged_ids.indices,
                                  values=weights_values,
                                  dense_shape=merged_ids.dense_shape)
  return merged_ids, merged_weights, row_counts, origin_shape_list

def coalesce_sparse_data(ids_list, weights_list, weight_type, format_rank=2):
  if all([w is None for w in weights_list]):
    weights_list = [None] * len(ids_list)
  else:
    for i, weights in enumerate(weights_list):
      if weights_list[i] is None:
        values = array_ops.ones_like(ids_list[i].values, dtype=weight_type)
        weights_list[i] = SparseTensor(indices=ids_list[i].indices,
                                       values=values,
                                       dense_shape=ids_list[i].dense_shape)
  ids_pair = [(ids, weights) for ids, weights in zip(ids_list, weights_list)]
  merged_ids, merged_weights, size_list, origin_shape_list = _safe_merge_sparse_tensor(ids_pair, format_rank=format_rank)
  return (merged_ids, merged_weights, size_list, origin_shape_list)

def add_to_collections(var, weight_collections):
  """Adds a var to the list of weight_collections provided.
  Handles the case for partitioned and non-partitioned variables.
  Args:
    var: A variable or Partitioned Variable.
    weight_collections: List of collections to add variable to.
  """
  for weight_collection in weight_collections:
    # The layer self.add_variable call already adds it to GLOBAL_VARIABLES.
    if weight_collection == ops.GraphKeys.GLOBAL_VARIABLES:
      continue
    if isinstance(var, variables.PartitionedVariable):
      for constituent_var in list(var):
        ops.add_to_collection(weight_collection, constituent_var)
    else:
      ops.add_to_collection(weight_collection, var)

def check_initializer_compatible(i1, i2):
  if not isinstance(i1, init_ops.Initializer):
    raise ValueError('i1 must be an Initializer object')
  if not isinstance(i2, init_ops.Initializer):
    raise ValueError('i2 must be an Initializer object')
  if i1.__class__ != i2.__class__:
    return False
  if i1.get_config() != i2.get_config():
    return False
  return True

def check_share_compatible(c1, c2):
  error = 'Cannot share HashTable with incompatible {}: {} vs {}'
  if c1.dimension != c2.dimension:
    return error.format('dimension', c1.dimension, c2.dimension)
  if c1.dtype != c2.dtype:
    return error.format('dtype', c1.dtype, c2.dtype)
  if not check_initializer_compatible(c1.initializer, c2.initializer):
    return 'Cannot share HashTable with incompatible initializer'
  if c1.trainable != c2.trainable:
    return error.format('trainable', c1.trainable, c2.trainable)
  return None

@contextlib.contextmanager
def merged_embedding_lookup_hook(hooks):
  if hooks is None:
    hooks = tuple()
  if hasattr(contextlib, "nested"):
    with contextlib.nested(*hooks):
      yield
  else:
    with contextlib.ExitStack() as stack:
      for hook in hooks:
        stack.enter_context(hook)
      yield
