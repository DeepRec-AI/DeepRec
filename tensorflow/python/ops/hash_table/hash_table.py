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
HashTable Variable.

@@SimpleHashTable
@@HashTable
@@DistributedHashTable
"""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_hash_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

@tf_export("hash_table.SimpleHashTable")
class SimpleHashTable(object):
  """TODO: Add DocString"""
  def __init__(
      self, distributed_name, concurrent_read, slicer=None,
      children=None, name=None):
    self._distributed_name = distributed_name
    self._children = [] if children is None else children
    with ops.name_scope(name, "SimpleHashTable") as name:
      handle_name = ops.name_from_scope_name(name)
      self._name = handle_name
      with ops.control_dependencies(None):
        self._handle = gen_hash_ops.hash_table_op(shared_name=handle_name, name=name)
        self._slicer = slicer
        with ops.colocate_with(self._handle):
          self._initializer = gen_hash_ops.hash_table_initialize_op(
              self._handle, True, concurrent_read, self._children,
              name="Initializer")
          self._false_initializer = gen_hash_ops.hash_table_initialize_op(
              self._handle, False, concurrent_read, self._children,
              name="FalseInitializer")

  @property
  def handle(self):
    return self._handle

  @property
  def initializer(self):
    return self._initializer

  @property
  def false_initializer(self):
    return self._false_initializer

  @property
  def slicer(self):
    return self._slicer

  @property
  def name(self):
    return self._name

  @property
  def op(self):
    return self._handle.op

  @property
  def device(self):
    return self._handle.device

  @property
  def graph(self):
    return self._handle.graph

  @property
  def distributed_name(self):
    return self._distributed_name

  @distributed_name.setter
  def distributed_name(self, distributed_name):
    self._distributed_name = distributed_name

  @property
  def children(self):
    return self._children

  def to_proto(v, export_scope=None):
    return None

  def from_proto(v, import_scope=None):
    return None

  def __lt__(self, other):
    if isinstance(other, SimpleHashTable):
      other = other._name
    return self._name < other

  def lookup(self, keys, admit_strategy=None, frequencies=None, name=None):
    if admit_strategy is None:
      with ops.colocate_with(self._handle):
        return gen_hash_ops.hash_table_lookup_op(
            self._handle, ops.convert_to_tensor(keys, dtype=dtypes.int64), name=name)
    else:
      if frequencies is None:
        freqs_tensor = constant_op.constant([], dtype=dtypes.int32)
      else:
        freqs_tensor = ops.convert_to_tensor(frequencies, dtype=dtypes.int32)
      with ops.colocate_with(self._handle):
        return gen_hash_ops.hash_table_lookup_with_admit_op(
            self._handle, ops.convert_to_tensor(keys, dtype=dtypes.int64),
            admit_strategy, freqs_tensor, name=name)

  def size(self, name=None):
    with ops.colocate_with(self._handle):
      return gen_hash_ops.hash_table_size_op(self._handle, name=name)

@tf_export("hash_table.HashTable")
class HashTable(object):
  """TODO: Add DocString"""
  DEFAULT_SLICE_SIZE = 4096

  def __init__(
      self, shape, dtype, distributed_name,
      initializer=None, init_func=None, segment_size=None,
      collections=None, trainable=True, slicer=None,
      hash_table=None, concurrent_read=True, children=None, name=None):
    self._distributed_name = distributed_name
    self._slots = {}
    self._concurrent_read = concurrent_read
    self._children = [] if children is None else children
    with ops.name_scope(name, "HashTable") as name:
      handle_name = ops.name_from_scope_name(name)
      self._name = handle_name
      if hash_table is None:
        hash_table = SimpleHashTable(
            distributed_name, concurrent_read, slicer, self._children)
      with ops.control_dependencies(None):
        with ops.colocate_with(hash_table.handle):
          if initializer == None and init_func == None:
            raise ValueError("initializer or initial_value must be specified.")
          if initializer != None and init_func != None:
            raise ValueError("initializer and initial_value must not be specified both.")

          if segment_size == None:
            segment_size = self.DEFAULT_SLICE_SIZE
          self._shape = tensor_shape.TensorShape(shape)
          self._dtype = dtypes.as_dtype(dtype)
          self._segment_shape = tensor_shape.TensorShape.concatenate(
              tensor_shape.TensorShape([segment_size]), self._shape)

          if initializer:
            if isinstance(initializer, type):
              initializer = initializer(self._dtype)
            init_func = lambda: initializer(shape=self._segment_shape, dtype=self._dtype, partition_info=None)

          self._hash_table = hash_table
          self._factory = function.Defun()(lambda: gen_hash_ops.copy_tensor(init_func()))
          self._handle = gen_hash_ops.tensible_variable_op(
              self._dtype, self._segment_shape, shared_name=handle_name, name=name)
          with ops.control_dependencies([hash_table.initializer]):
            self._initializer = self.initializer_without_hashtable(True, "Initializer")
            self._false_initializer = self.initializer_without_hashtable(False, "FalseInitializer")
    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          "collections argument to Variable constructor must be a list, tuple, "
          "or set. Got %s of type %s" % (collections, type(collections)))
    trainable = trainable if trainable is not None else True
    if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
      collections = list(collections) + [ops.GraphKeys.TRAINABLE_VARIABLES]
    ops.add_to_collections(collections, self)

  def __repr__(self):
    return "<tf.hash_table.HashTable '%s' shape=%s dtype=%s" % (
        self.name, self.shape, self.dtype.name)

  @property
  def handle(self):
    return self._handle

  @property
  def initializer(self):
    return self._initializer

  @property
  def false_initializer(self):
    return self._false_initializer

  @property
  def hash_table(self):
    return self._hash_table

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return self._shape

  @property
  def name(self):
    return self._name

  @property
  def op(self):
    return self._handle.op

  @property
  def device(self):
    return self._handle.device

  @property
  def graph(self):
    return self._handle.graph

  @property
  def slicer(self):
    return self._hash_table._slicer

  @slicer.setter
  def slicer(self, slicer):
    self._hash_table._slicer = slicer

  @property
  def distributed_name(self):
    return self._distributed_name

  @property
  def slots(self):
    return self._slots

  @property
  def children(self):
    return self._children

  def get_slot(self, name):
    return self._slots[name]

  @property
  def snapshot(self):
    return gen_hash_ops.hash_table_snapshot_op(self.hash_table.handle)

  @distributed_name.setter
  def distributed_name(self, distributed_name):
    self._distributed_name = distributed_name

  def get_shape(self):
    return self._shape

  def initializer_without_hashtable(self, initialized=True, name=None):
    with ops.name_scope(name, "Initializer") as name:
      init_op = gen_hash_ops.tensible_variable_initialize_op(
          self._handle, self._hash_table.handle, self._dtype,
          self._segment_shape, self._factory, initialized,
          name="Initializer")
      return init_op

  def initialize_resource_op(self, name=None):
    with ops.control_dependencies([self._hash_table.initializer]):
      return gen_hash_ops.tensible_variable_initialize_op(
          self._handle, self._hash_table.handle, self._dtype,
          self._segment_shape, self._factory)

  def lookup(self, keys, admit_strategy=None, frequencies=None, default_value=0, name=None):
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "tensible_variable_lookup") as name:
        ids = self.gen_ids(keys, admit_strategy, frequencies)
        return self.lookup_by_id(ids, default_value, name)

  def gen_ids(self, keys, admit_strategy=None, frequencies=None, name=None):
    with ops.colocate_with(self._handle):
      return self._hash_table.lookup(keys, admit_strategy, frequencies, name)

  def lookup_by_id(self, ids, default_value=0, name=None):
    default_value = ops.convert_to_tensor(default_value, dtype=self._dtype)
    with ops.colocate_with(self._handle):
      return gen_hash_ops.tensible_variable_gather(
          self._handle, ids, default_value, name=name)

  def size(self, name=None):
    return self._hash_table.size(name)

  def get_or_create_slot(
      self, shape, dtype, distributed_name, initializer=None, init_func=None,
      segment_size=None, collections=None, trainable=False, name=None):
    if distributed_name in self._slots:
      return self._slots[distributed_name]
    return self.create_slot(
        shape, dtype, distributed_name, initializer, init_func, segment_size,
        collections, trainable, name)

  def create_slot(
      self, shape, dtype, distributed_name, initializer=None, init_func=None, segment_size=None,
      collections=None, trainable=False, name=None):
    if distributed_name in self._slots:
      k = 1
      while "{}_{}".format(distributed_name, k) in self._slots:
        k += 1
      distributed_name = "{}_{}".format(distributed_name, k)
    slot = HashTable(
      shape, dtype, distributed_name,
      initializer, init_func, segment_size, collections, trainable,
      None, self._hash_table, self._concurrent_read, self._children, name)
    self._slots[distributed_name] = slot
    return slot

  def to_proto(v, export_scope=None):
    return None

  def from_proto(v, import_scope=None):
    return None

  def __lt__(self, other):
    if isinstance(other, HashTable):
      other = other._name
    return self._name < other

  def scatter_update(self, keys, values, name=None):
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "tensible_scatter_update") as name:
        return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_update, name=name)

  def scatter_add(self, keys, values, name=None):
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "tensible_scatter_add") as name:
        return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_add, name=name)

  def scatter_sub(self, keys, values, name=None):
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "tensible_scatter_sub") as name:
        return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_sub)

  def scatter_mul(self, keys, values, name=None):
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "tensible_scatter_mul") as name:
        return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_mul)

  def scatter_div(self, keys, values, name=None):
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "tensible_scatter_div") as name:
        return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_div)

  def _scatter_op(self, keys, values, scatter_func, name=None):
    return scatter_func(self, self.gen_ids(keys), values, name=name)

@tf_export('hash_table.FixedSizeHashTablePartitioner')
class FixedSizeHashTablePartitioner(object):
  def __init__(self, part_num):
    self._part_num = part_num

  def __call__(self, slice_size):
    slicer = []
    x = 0
    for i in range(self._part_num):
      slicer.append(x)
      x += slice_size // self._part_num
      if i < slice_size % self._part_num:
        x += 1
    return slicer

@tf_export("hash_table.DistributedHashTable")
class DistributedHashTable(object):
  """TODO: Add DocString"""
  _DEFAULT_SLICER_SIZE = 65536
  def __init__(
      self, shape, dtype,
      initializer=None, init_func=None, segment_size=None,
      collections=None, trainable=True, partitioner=None,
      slice_size=None, slicer=None, hash_tables=None, concurrent_read=True,
      children=None, name=None):
    if slice_size is None:
      slice_size = DistributedHashTable._DEFAULT_SLICER_SIZE

    if slicer is None and partitioner is None:
      raise ValueError('must specify slicer or partitioner')
    if slicer is None:
      slicer = partitioner(slice_size)

    self._children = [] if children is None else children
    if hash_tables is None:
      with ops.name_scope(name, "DistributedHashTable") as name:
        distributed_name = ops.name_from_scope_name(name)
        oslicer = slicer + [slice_size]
        hash_tables = []
        for i in range(len(slicer)):
          hash_tables.append(
            HashTable(shape, dtype, distributed_name, initializer,
                      init_func, segment_size, collections, trainable,
                      [oslicer[i], oslicer[i + 1], slice_size], None,
                      concurrent_read, children, "HashTable_" + str(i)))

    if len(hash_tables) != len(slicer):
      raise RuntimeError("HashTable size should be equal to slicer size")

    self._shape = shape
    self._dtype = dtype
    self._slice_size = slice_size
    self._slicer = slicer
    self._hash_tables = hash_tables
    self._name = hash_tables[0].distributed_name

  @property
  def dtype(self):
    return self._dtype

  @property
  def shape(self):
    return self._shape

  @property
  def partitions(self):
    return list(self._hash_tables)

  @property
  def initializer(self):
    return [i.initializer for i in self._hash_tables]

  @property
  def device(self):
    return [i.device for i in self._hash_tables]

  @property
  def name(self):
    return self._name

  @property
  def snapshot(self):
    keys = []
    ids = []
    for partition in self.partitions:
      key, id = partition.snapshot
      keys.append(key)
      ids.append(id)
    return [array_ops.concat(keys, 0), array_ops.concat(ids, 0)]

  def initializer_without_hashtable(self, name=None):
    with ops.name_scope(name, "DistributedHashTable_Initializer") as name:
      return [i.initializer_without_hashtable() for i in self._hash_tables]

  def lookup(self, keys, admit_strategy_factory=None, frequencies=None, default_value=0, name=None):
    with ops.name_scope(name, "DistributedHashTable_Lookup") as name:
      keys_per_device, gather_ids_per_device, indices_per_device, dense_shape =\
          self.gen_ids(keys, admit_strategy_factory, frequencies)
      return self.lookup_by_id(gather_ids_per_device, indices_per_device, dense_shape, default_value)

  def gen_ids(self, keys, admit_strategy_factory=None, frequencies=None, name=None):
    if admit_strategy_factory is None:
        admit_strategy_factory = lambda ht: None
    keys = ops.convert_to_tensor(keys, dtype=dtypes.int64)
    with ops.name_scope(name, "DistributedHashTable_GenIds") as name:
      flat_keys = array_ops.reshape(keys, [-1])
      original_indices = math_ops.range(array_ops.size(flat_keys))
      part_id = gen_hash_ops.hash_slice(self._slicer, flat_keys, self._slice_size)
      keys_per_device = data_flow_ops.dynamic_partition(flat_keys, part_id, len(self._slicer))
      if frequencies is None:
        freqs_per_device = [constant_op.constant([], dtype=dtypes.int32)] * len(self._slicer)
      else:
        flat_freqs = array_ops.reshape(frequencies, [-1])
        freqs_per_device= data_flow_ops.dynamic_partition(flat_freqs, part_id, len(self._slicer))
      indices_per_device = data_flow_ops.dynamic_partition(original_indices, part_id, len(self._slicer))
      gather_ids_per_device = [self._hash_tables[i].gen_ids(keys_per_device[i], admit_strategy_factory(self._hash_tables[i]), freqs_per_device[i])
                               for i in range(len(keys_per_device))]
      return keys_per_device, gather_ids_per_device, indices_per_device, array_ops.shape(keys)

  def split_values_by_key(self, keys, values, name=None):
    keys = ops.convert_to_tensor(keys, dtype=dtypes.int64)
    with ops.name_scope(name, "DistributedHashTable_SplitValues") as name:
      flat_keys = array_ops.reshape(keys, [-1])
      part_id = gen_hash_ops.hash_slice(self._slicer, flat_keys, self._slice_size)
      values_per_device = data_flow_ops.dynamic_partition(values, part_id, len(self._slicer))
    return values_per_device

  def lookup_by_id(self, gather_ids_per_device, indices_per_device, dense_shape, default_value=0, name=None):
    with ops.name_scope(name, "DistributedHashTable_LookupById") as name:
      values = [self._hash_tables[i].lookup_by_id(gather_ids_per_device[i], default_value)
                for i in range(len(gather_ids_per_device))]
      ret = data_flow_ops.parallel_dynamic_stitch(
            indices_per_device, values, name=name)
      return array_ops.reshape(ret,
                              array_ops.concat(
                                  [dense_shape, self._shape], 0))

  def size(self, name=None):
    with ops.name_scope(name, "DistributedHashTable_Size") as name:
      return math_ops.add_n([hash_table.size() for hash_table in self._hash_tables])

  def create_slot(
      self, shape, dtype, initializer=None, init_func=None, segment_size=None,
      collections=None, trainable=False, name=None):
    with ops.name_scope(name, "slot") as name:
      distributed_name = '{}/slots/{}'.format(self._name, name)
      hash_tables = [
          i.create_slot(
            shape, dtype, distributed_name, initializer, init_func,
            segment_size, collections, trainable) for i in self._hash_tables]
      return DistributedHashTable(shape, dtype, hash_tables=hash_tables,
                                  slice_size=self._slice_size, slicer=self._slicer)

  def to_proto(v, export_scope=None):
    return None

  def from_proto(v, import_scope=None):
    return None

  def __lt__(self, other):
    if isinstance(other, DistributedHashTable):
      other = other._name
    return self._name < other

  def scatter_update(self, keys, values, name=None):
    with ops.name_scope(name, "DistributedHashTable_Scatter_Update") as name:
      return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_update, name=name)

  def scatter_add(self, keys, values, name=None):
    with ops.name_scope(name, "DistributedHashTable_Scatter_Add") as name:
      return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_add, name=name)

  def scatter_sub(self, keys, values, name=None):
    with ops.name_scope(name, "DistributedHashTable_Scatter_Sub") as name:
      return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_sub, name=name)

  def scatter_mul(self, keys, values, name=None):
    with ops.name_scope(name, "DistributedHashTable_Scatter_Mul") as name:
      return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_mul, name=name)

  def scatter_div(self, keys, values, name=None):
    with ops.name_scope(name, "DistributedHashTable_Scatter_Div") as name:
      return self._scatter_op(keys, values, gen_hash_ops.tensible_variable_scatter_div, name=name)

  def _scatter_op(self, keys, values, scatter_func, name=None):
    _, ids_per_device, _, _ = self.gen_ids(keys, name=name)
    values_per_device = self.split_values_by_key(keys, values, name=name)
    update_ops = [scatter_func(self.partitions[i], ids_per_device[i], values_per_device[i])
        for i in range(len(ids_per_device))]
    return control_flow_ops.group(update_ops)

@ops.RegisterGradient("TensibleVariableGather")
def _GatherGrad(op, grad):
  """Gradient for gather op."""
  # Build appropriately shaped IndexedSlices
  handle = op.inputs[0]
  indices = op.inputs[1]
  params_shape = handle.get_shape()
  return (ops.IndexedSlices(grad, indices, params_shape), None, None)

@tf_export("hash_table.HashTableKeyMapperFactory")
class HashTableKeyMapperFactory(object):
  # user should override this function
  def __call__(self, hash_table):
    # the map function should return a tensor of keys
    # return None for error occations
    def map(keys, ids):
      raise RuntimeError("HashTableKeyMapper map method not implemeted")
    return map

def _hashtable_to_tensor(var, dtype=None, name=None, as_ref=False):
  return var.handle


ops.register_tensor_conversion_function(HashTable, _hashtable_to_tensor)

_HASH_TABLE_RESTORE_CLEAR = True

@tf_export("hash_table.restore_without_clear_scope")
@contextlib.contextmanager
def restore_without_clear_scope():
  """Set clear flag when restoring HashTable.

  Coalesced hash table use this flag to determine whether clear current
  internal data when running a restore op. Normally, this flag is always True
  for manually restoring or failover recovering. However, it should set to
  False when consecutively restoring from mutiple checkpoints(ModelBank).
  """
  global _HASH_TABLE_RESTORE_CLEAR
  old_flag = _HASH_TABLE_RESTORE_CLEAR
  _HASH_TABLE_RESTORE_CLEAR = False
  yield
  _HASH_TABLE_RESTORE_CLEAR = old_flag

def restore_clear():
  """Get hash table clear flag"""
  global _HASH_TABLE_RESTORE_CLEAR
  return _HASH_TABLE_RESTORE_CLEAR

