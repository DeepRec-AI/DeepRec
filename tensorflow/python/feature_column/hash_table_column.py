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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import json
import math

from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.feature_column import coalesced_utils
from tensorflow.python.feature_column.feature_column_v2 import DenseColumn, WeightedCategoricalColumn, CutoffCategoricalColumn
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.hash_table import embedding
from tensorflow.python.ops.hash_table import hash_table
from tensorflow.python.util.tf_export import tf_export

@tf_export('feature_column.hash_table_column')
def hash_table_column(
    categorical_column, dimension, dtype=None, initializer=None, combiner=None,
    partitioner=None, trainable=True, embedding_lookup_hooks=(),
    coalesced_scope=None):
  """`DenseColumn` that converts from sparse, categorical input.

  Args:
    categorical_column: An `_CategoricalColumn` instance created by a
      `categorical_column_with_*` function. It produces the sparse Ids
      and weights that inputs to the emebdding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    dtype: defines the type of embedding, defaults to tf.float32
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently 'mean', 'sqrtn', 'sum' and 'tile' are
      supported, with 'mean' the default. 'sqrtn' often achieves good accuracy,
      in particular with bag-of-words columns. Each of this can be thought as
      example level normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    partitioner: A function accepts one int agument that defines how to divide
      Ids to multiple device, defaults to FixedSizeHashTablePartitioner(1)
    trainable: defines whether the emebdding need to update, defaults to True
    embedding_lookup_hooks: Hashtable hooks when look up, defaults to empty
      tuple

  Returns:
    A _HashTableColumn object that converts sparse ids to embedding

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if initializer is not a callable
  """
  if (dimension is None) or (dimension < 1):
    raise ValueError("Invalid dimension {}.".format(dimension))
  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. '
                     'HashTable of column_name: {}'.format(
                         categorical_column.name))
  if dtype is None:
    dtype = dtypes.float32
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))
  if combiner is None:
    combiner = 'mean'
  if embedding_lookup_hooks is None:
    embedding_lookup_hooks = ()
  if not isinstance(embedding_lookup_hooks, (list, tuple)):
    embedding_lookup_hooks = (embedding_lookup_hooks, )
  embedding_lookup_hooks = tuple(embedding_lookup_hooks)
  if partitioner is None:
    partitioner = hash_table.FixedSizeHashTablePartitioner(1)
  if coalesced_scope is None:
    coalesced_scope = current_coalesced_scope()
  column = HashTableColumn(
      categorical_column=categorical_column,
      dimension=dimension,
      dtype=dtype,
      initializer=initializer,
      combiner=combiner,
      partitioner=partitioner,
      trainable=trainable,
      embedding_lookup_hooks=embedding_lookup_hooks,
      coalesced_scope=coalesced_scope)
  if coalesced_scope:
    coalesced_scope.add_column(column)
  return column

class HashTableColumn(
    DenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'HashTableColumn',
        ('categorical_column', 'dimension', 'dtype', 'initializer',
         'combiner', 'partitioner', 'trainable',
         'embedding_lookup_hooks', 'coalesced_scope'))):

  @property
  def _is_v2_column(self):
    return (isinstance(self.categorical_column, FeatureColumn) and
            self.categorical_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_hash_table'.format(self.categorical_column.name)

  @property
  def table_name(self):
    c = self.categorical_column
    while isinstance(c, (WeightedCategoricalColumn, CutoffCategoricalColumn)):
      c = c.categorical_column
    return c.name

  @property
  def var_scope_name(self):
    return self.table_name

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  def transform_feature(self, transformation_cache, state_manager):
    """Transforms input data."""
    return transformation_cache.get(self.categorical_column, state_manager)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape([self.dimension])

  def create_state(self, state_manager):
    """Creates HashTable handle"""
    if self.coalesced_scope:
      self.coalesced_scope.create_state_by_column(self)
    else:
      state_manager.create_hashtable(self,
                                      name=self.table_name,
                                      shape=[self.dimension],
                                      dtype=self.dtype,
                                      initializer=self.initializer,
                                      partitioner=self.partitioner,
                                      trainable=self.trainable)

  def _get_dense_tensor_internal_helper(self, sparse_tensors,
                                        embedding_weights):
    sparse_ids = sparse_tensors.id_tensor
    sparse_weights = sparse_tensors.weight_tensor
    with coalesced_utils.merged_embedding_lookup_hook(self.embedding_lookup_hooks):
      return embedding_ops.safe_embedding_lookup_sparse(
          embedding_weights=embedding_weights,
          sparse_ids=sparse_ids,
          sparse_weights=sparse_weights,
          combiner=self.combiner,
          name='%s_weights' % self.name)

  def _get_dense_tensor_internal(self, sparse_tensors, state_manager):
    """Private method that follows the signature of get_dense_tensor."""
    embedding_weights = state_manager.get_variable(
        self, name=self.table_name)
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns embedding tensor after doing the sparse segment combine
    """
    if self.coalesced_scope:
      return self.coalesced_scope.get_dense_tensor_by_column_v2(
          self, transformation_cache, state_manager)

    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._get_dense_tensor_internal(sparse_tensors, state_manager)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

@tf_export('feature_column.shared_hash_table_columns')
def shared_hash_table_columns(
    categorical_columns, dimension, dtype=None, initializer=None,
    combiner=None, partitioner=None, trainable=True, embedding_lookup_hooks=(),
    shared_name=None, coalesced_scope=None):
  """List of dense columns that convert from sparse, categorical input.

  This is similar to 'hash_table_column', except that it produces a list of
  embedding columns that share the same hash table.

  see hash_table_column and shared_embedding_column.

  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    dtype: defines the type of embedding, defaults to tf.float32
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently 'mean', 'sqrtn', 'sum' and 'tile' are
      supported, with 'mean' the default. 'sqrtn' often achieves good accuracy,
      in particular with bag-of-words columns. Each of this can be thought as
      example level normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    partitioner: A function accepts one int agument that defines how to divide
      Ids to multiple device, defaults to FixedSizeHashTablePartitioner(1)
    trainable: defines whether the emebdding need to update, defaults to True
    embedding_lookup_hooks: Hashtable hooks when look up, defaults to empty
      tuple
    shared_name: Optional name of the collection where
      shared hash table are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared hash table.

  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('shared_hash_table_columns are not supported when eager '
                       'execution is enabled.')
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if dtype is None:
    dtype = dtypes.float32
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))
  if combiner is None:
    combiner = 'mean'
  if partitioner is None:
    partitioner = hash_table.FixedSizeHashTablePartitioner(1)
  if embedding_lookup_hooks is None:
    embedding_lookup_hooks = ()
  if not isinstance(embedding_lookup_hooks, (list, tuple)):
    embedding_lookup_hooks = (embedding_lookup_hooks,)
  embedding_lookup_hooks = tuple(embedding_lookup_hooks)
  if not shared_name:
    shared_name = '_'.join(c.name for c in sorted_columns)
    shared_name += '_shared_hash_table'
  if coalesced_scope is None:
    coalesced_scope = current_coalesced_scope()
  result = []
  for column in categorical_columns:
    shared_column = SharedHashTableColumn(categorical_column=column,
                                          dimension=dimension,
                                          dtype=dtype,
                                          initializer=initializer,
                                          combiner=combiner,
                                          partitioner=partitioner,
                                          trainable=trainable,
                                          embedding_lookup_hooks=embedding_lookup_hooks,
                                          coalesced_scope=coalesced_scope,
                                          shared_name=shared_name)
    if coalesced_scope:
      coalesced_scope.add_column(shared_column)
    result.append(shared_column)
  return result

@tf_export('feature_column.shared_hash_table_column')
def shared_hash_table_column(
    categorical_column, dimension, shared_name, dtype=None, initializer=None,
    combiner=None, partitioner=None, trainable=True, embedding_lookup_hooks=(),
    coalesced_scope=None):
  """Create shared hash table column with multiple single calls

  See `shared_hash_table_columns`
  """
  if context.executing_eagerly():
    raise RuntimeError('shared_hash_table_columns are not supported when eager '
                       'execution is enabled.')
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified.')
  if dtype is None:
    dtype = dtypes.float32
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))
  if combiner is None:
    combiner = 'mean'
  if embedding_lookup_hooks is None:
    embedding_lookup_hooks = ()
  if not isinstance(embedding_lookup_hooks, (list, tuple)):
    embedding_lookup_hooks = (embedding_lookup_hooks,)
  embedding_lookup_hooks = tuple(embedding_lookup_hooks)
  if partitioner is None:
    partitioner = hash_table.FixedSizeHashTablePartitioner(1)
  if coalesced_scope is None:
    coalesced_scope = current_coalesced_scope()
  shared_column = SharedHashTableColumn(categorical_column=categorical_column,
                                        dimension=dimension,
                                        dtype=dtype,
                                        initializer=initializer,
                                        combiner=combiner,
                                        partitioner=partitioner,
                                        trainable=trainable,
                                        embedding_lookup_hooks=embedding_lookup_hooks,
                                        coalesced_scope=coalesced_scope,
                                        shared_name=shared_name)
  if coalesced_scope:
    coalesced_scope.add_column(shared_column)
  return shared_column

class SharedHashTableColumn(
    DenseColumn,
    fc_old._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        'SharedHashTableColumn',
        ('categorical_column', 'dimension', 'dtype', 'initializer',
         'combiner', 'partitioner', 'trainable', 'embedding_lookup_hooks',
         'coalesced_scope', 'shared_name'))):
  """See `hash_table_column`."""

  @property
  def _is_v2_column(self):
    return (isinstance(self.categorical_column, FeatureColumn) and
            self.categorical_column._is_v2_column)  # pylint: disable=protected-access

  @property
  def name(self):
    """See `FeatureColumn` base class."""
    return '{}_shared_hash_table'.format(self.categorical_column.name)

  @property
  def table_name(self):
    return self.shared_name

  @property
  def var_scope_name(self):
    return self.table_name

  @property
  def embedding_name(self):
    return self.table_name

  @property
  def parse_example_spec(self):
    """See `FeatureColumn` base class."""
    return self.categorical_column.parse_example_spec

  def transform_feature(self, transformation_cache, state_manager):
    """Transforms input data."""
    return transformation_cache.get(self.categorical_column, state_manager)

  @property
  def variable_shape(self):
    """See `DenseColumn` base class."""
    return tensor_shape.TensorShape([self.dimension])

  def get_embedding(self):
    shared_hash_table_collection = ops.get_collection(self.shared_name + "_hashtable")
    if shared_hash_table_collection:
      if len(shared_hash_table_collection) > 1:
        raise ValueError(
            'Collection {} can only contain one variable. '
            'Suggested fix A: Choose a unique name for this collection. '
            'Suggested fix B: Do not add any variables to this collection. '
            'The feature_column library already adds a variable under the '
            'hood.'.format(shared_hash_table_collection))
      table = shared_hash_table_collection[0][0]
      err_msg = coalesced_utils.check_share_compatible(
          self, shared_hash_table_collection[0][1])
      if err_msg:
        raise ValueError(err_msg)
      return table
    else:
      raise ValueError("Embedding not created yet.")

  def create_state(self, state_manager):
    """Creates HashTable handle"""
    if self.coalesced_scope:
      self.coalesced_scope.create_state_by_column(self)
    else:
      shared_hash_table_collection = ops.get_collection(self.shared_name + "_hashtable")
      if shared_hash_table_collection:
        if len(shared_hash_table_collection) > 1:
          raise ValueError(
              'Collection {} can only contain one variable. '
              'Suggested fix A: Choose a unique name for this collection. '
              'Suggested fix B: Do not add any variables to this collection. '
              'The feature_column library already adds a variable under the '
              'hood.'.format(shared_hash_table_collection))
      else:
        state_manager.create_hashtable(self,
                                       name=self.table_name,
                                       shape=[self.dimension],
                                       dtype=self.dtype,
                                       initializer=self.initializer,
                                       partitioner=self.partitioner,
                                       trainable=self.trainable)
        table = state_manager.get_variable(
            self, name=self.table_name)
        ops.add_to_collection(self.shared_name + "_hashtable", (table, self))

  def _get_dense_tensor_internal_helper(self, sparse_tensors,
                                        embedding_weights):
    sparse_ids = sparse_tensors.id_tensor
    sparse_weights = sparse_tensors.weight_tensor

    # Return embedding lookup result.
    with coalesced_utils.merged_embedding_lookup_hook(self.embedding_lookup_hooks):
      return embedding_ops.safe_embedding_lookup_sparse(
          embedding_weights=embedding_weights,
          sparse_ids=sparse_ids,
          sparse_weights=sparse_weights,
          combiner=self.combiner,
          name='%s_weights' % self.name)

  def _get_dense_tensor_internal(self, sparse_tensors, state_manager):
    """Private method that follows the signature of get_dense_tensor."""
    embedding_weights = self.get_embedding()
    return self._get_dense_tensor_internal_helper(sparse_tensors,
                                                  embedding_weights)

  def get_dense_tensor(self, transformation_cache, state_manager):
    """Returns embedding tensor after doing the sparse segment combine
    """
    if self.coalesced_scope:
      return self.coalesced_scope.get_dense_tensor_by_column_v2(
          self, transformation_cache, state_manager)

    sparse_tensors = self.categorical_column.get_sparse_tensors(
        transformation_cache, state_manager)
    return self._get_dense_tensor_internal(sparse_tensors, state_manager)

  @property
  def parents(self):
    """See 'FeatureColumn` base class."""
    return [self.categorical_column]

_global_scopes = []

@tf_export('feature_column.coalesced_hash_table_scope')
@contextlib.contextmanager
def coalesced_hash_table_scope(name=None):
  global _global_scopes
  scope = CoalescedScope(name=name)
  _global_scopes.append(scope)
  yield
  scope = _global_scopes.pop()
  scope.build()

def current_coalesced_scope():
  global _global_scopes
  return None if len(_global_scopes) == 0 else _global_scopes[-1]

@tf_export('feature_column.CoalescedScope')
class CoalescedScope(coalesced_utils.CoalescedScopeBase):
  def __init__(self, name=None):
    if name is None:
      name = 'CoalescedHashTable'
    super(CoalescedScope, self).__init__(name)

  def allowed_column_types(self):
    return CoalescedHashTableColumn._COALESCING_TYPES

  def build(self):
    if self._built:
      return
    cluster = collections.defaultdict(list)
    for name, column in self._columns.items():
      h = coalesced_utils.make_cluster_signature(column, True)
      cluster[h].append((name, column))

    for h, names_and_columns in cluster.items():
      names_and_columns.sort(key=lambda x: x[1].table_name)

    for h, names_and_columns in cluster.items():
      names, columns = zip(*names_and_columns)
      coalesced_column = CoalescedHashTableColumn(columns, self)
      for name in names:
        self._coalesced_map[name] = coalesced_column
    self._built = True

MASK_LENGTH = 12
MASK_REMAIN_LENGTH = 64 - MASK_LENGTH
MASK = (1 << MASK_REMAIN_LENGTH) - 1

class CoalescedHashTableColumn(object):
  """Coalescing _HashTableColumns into one according to signature.

  Args:
    columns: An iterable containing the _HashTableColumns that would be
      coalesced according the signature of the HashTable.
    scope: An CoalescedScope object that this column belongs to
  Raises:
    ValueError: if an item in `columns` is not a `HashTableColumn` or
      `SharedHashTableColumn`
    ValueError: if signature of columns not compatible
  """
  _COALESCING_TYPES = (HashTableColumn, SharedHashTableColumn)

  def __init__(self, columns, scope=None):
    for i, c in enumerate(columns):
      if not isinstance(c, CoalescedHashTableColumn._COALESCING_TYPES):
        raise ValueError('columns must be a list of HashTableColumn, '
                         'Given {} at index {}'.format(c, i))
    if len(columns) == 0:
      raise ValueError('columns cannot be empty')

    coalesced_utils.check_coalesced_columns_compatible(columns, True)

    self._columns = columns
    self._runtime_columns = collections.defaultdict(list)
    self.build_runtime_columns(columns)
    self._scope = scope

    self._column_index = {}
    self._children = []
    self._column = columns[0]
    i = 0
    for c in self._columns:
      name = c.table_name
      if name not in self._column_index:
        self._column_index[name] = i
        self._children.append(name)
        i += 1

  @property
  def columns(self):
    return self._columns

  def build_runtime_columns(self, columns):
    for i, column in enumerate(columns):
      h = coalesced_utils._make_runtime_signature(column, True)
      self._runtime_columns[h].append((i, column))

  def encode(self, column, data):
    if not isinstance(column, CoalescedHashTableColumn._COALESCING_TYPES):
      raise ValueError("column should be a HashTableColumn, "
                      "but got: {}".format(column))
    if not isinstance(data, SparseTensor):
      raise ValueError("data should be a SparseTensor, "
                       "but got: {}".format(data))
    name = column.table_name
    if name not in self._column_index:
      raise ValueError("HashTableColumn {} is not a child".format(column.name))
    index = self._column_index[name] << MASK_REMAIN_LENGTH
    values = data.values
    values = bitwise_ops.bitwise_and(
        values, constant_op.constant(MASK, dtypes.int64))
    values = bitwise_ops.bitwise_or(
        values, constant_op.constant(index, dtypes.int64))
    return SparseTensor(indices=data.indices, values=values,
                        dense_shape=data.dense_shape)

  def make_sparse_inputs(self, transformation_cache, state_manager):
    """Transforms input data.
    Merge multiple input (ids, weights) to one, encoding with index of column.
    """
    result_list = []
    for runtime_columns in self._runtime_columns.values():
      ids_list = []
      weights_list = []
      weight_type = None
      for c in runtime_columns:
        sparse_tensors = c[1].categorical_column.get_sparse_tensors(
            transformation_cache, state_manager)
        ids, weights = fc_utils.parse_sparse_data(sparse_tensors)
        if ids is None:
          raise ValueError("sparse ids cannot be None")
        ids_list.append(self.encode(c[1], ids))
        weights_list.append(weights)
        if weights is not None:
          if weight_type is None:
            weight_type = weights.dtype
          elif weight_type != weights.dtype:
            raise ValueError('all weights should have same dtype, but got '
                             '{} and {}'.format(weight_type, weights.dtype))
      if weight_type is None:
        weight_type = dtypes.float32
      result = coalesced_utils.coalesce_sparse_data(ids_list, weights_list, weight_type)
      result_list.append(result)
    return result_list

  def get_or_create_embedding_weights(self):
    if not hasattr(self, '_embedding_weights'):
      name = self._scope.get_name() if self._scope else 'CoalescedHashTable'
      table = variable_scope.get_hash_table(
          name, [self._column.dimension], dtype=self._column.dtype,
          initializer=self._column.initializer,
          partitioner=self._column.partitioner,
          trainable=self._column.trainable,
          children=self._children)
      self._embedding_weights = table
    return self._embedding_weights

  def _embedding_lookup_sparse(self, table, ids, weights, combiner, embedding_lookup_hooks):
    with coalesced_utils.merged_embedding_lookup_hook(embedding_lookup_hooks):
      return embedding_ops.safe_embedding_lookup_sparse(table,
                                                        ids,
                                                        weights,
                                                        combiner=combiner,
                                                        prune=False)

  def get_dense_tensor(self, transformation_cache, state_manager):
    hash_table = self.get_or_create_embedding_weights()
    lookup_input_list = self.make_sparse_inputs(
        transformation_cache, state_manager)
    embedding_outputs = []
    for lookup_input, cids_and_columns in zip(*(lookup_input_list,
                                       self._runtime_columns.values())):
      cids, columns = zip(*cids_and_columns)
      embeddings = self._embedding_lookup_sparse(
          hash_table, lookup_input[0], lookup_input[1],
          combiner=columns[0].combiner,
          embedding_lookup_hooks=columns[0].embedding_lookup_hooks)
      values = array_ops.split(embeddings, lookup_input[2])
      results = []
      for value, origin_shape, col in zip(values, lookup_input[3], columns):
        origin_rank = array_ops.size(origin_shape)
        if col.combiner == 'tile':
          real_dim = array_ops.gather(origin_shape, origin_rank - 1) * self._column.dimension
          value = array_ops.slice(value,
                                 [0, 0],
                                 [-1, real_dim])
        value = array_ops.reshape(
            value,
            array_ops.concat([
                array_ops.slice(origin_shape, [0], [origin_rank - 1]),
                array_ops.slice(array_ops.shape(value), [1], [-1])
            ], 0))
        results.append(value)
      embedding_outputs.extend(zip(cids, results))
    return list(zip(*sorted(embedding_outputs)))[1]
