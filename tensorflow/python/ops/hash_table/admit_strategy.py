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
HashTable AdmitStrategy.

@@BloomFilterAdmitStrategy
@@DistributedBloomFilterAdmitStrategy
"""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_hash_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

@tf_export("hash_table.BloomFilterAdmitStrategy")
class BloomFilterAdmitStrategy(object):
  """TODO: Add DocString"""
  _DEFAULT_ELEMENT_SIZE = 2 ** 20
  _DEFAULT_FALSE_POSITIVE_PROBABILITY = 0.01
  _DEFAULT_SLICE_OFFSET = 0
  _DEFAULT_SLICE_SIZE = 1
  def __init__(self,
               minimum_frequency,
               max_element_size=None,
               false_positive_probability=None,
               slicer=None,
               hash_table=None,
               distributed_name=None,
               collections=None,
               name=None):
    self._minimum_frequency = minimum_frequency
    if max_element_size is None:
      max_element_size = BloomFilterAdmitStrategy._DEFAULT_ELEMENT_SIZE
    self._max_element_size = max_element_size

    if false_positive_probability is None:
      false_positive_probability = BloomFilterAdmitStrategy._DEFAULT_FALSE_POSITIVE_PROBABILITY
    self._false_positive_probability= false_positive_probability

    if slicer is None:
      self._slice_offset = BloomFilterAdmitStrategy._DEFAULT_SLICE_OFFSET
      self._max_slice_size = BloomFilterAdmitStrategy._DEFAULT_SLICE_SIZE
      self._slice_size = self._max_slice_size - self._slice_offset
    else:
      self._slice_offset = slicer[0]
      self._max_slice_size = slicer[-1]
      self._slice_size = slicer[1] - slicer[0]

    self._hash_table = hash_table

    self._distributed_name = distributed_name

    size_per_slice = (max_element_size + self._max_slice_size - 1) // self._max_slice_size
    self._bucket_size = self._calc_bucket_size(size_per_slice,
        false_positive_probability)
    self._shape = tensor_shape.TensorShape([self._slice_size, self._bucket_size])
    self._dtype = self._optimal_dtype(minimum_frequency)

    self._num_hash_func = self._calc_hash_func_num(false_positive_probability)

    with ops.name_scope(name, "BloomFilter") as name:
      handle_name = ops.name_from_scope_name(name)
      self._name = handle_name
      with ops.control_dependencies(None):
        with ops.device(None if hash_table is None else hash_table.device):
          self._handle = gen_hash_ops.bloom_filter_admit_strategy_op(
              shared_name=handle_name, name=name)
          self._initializer = gen_hash_ops.bloom_filter_initialize_op(
              self._handle, min_frequency=minimum_frequency,
              num_hash_func=self._num_hash_func, slice_offset=self._slice_offset,
              max_slice_size=self._max_slice_size, dtype=self._dtype,
              shape=self._shape, initialized=True, name="Initializer")
          self._false_initializer = gen_hash_ops.bloom_filter_initialize_op(
              self._handle, min_frequency=minimum_frequency,
              num_hash_func=self._num_hash_func, slice_offset=self._slice_offset,
              max_slice_size=self._max_slice_size, dtype=self._dtype,
              shape=self._shape, initialized=False, name="FalseInitializer")

    if collections is None:
      collections = [ops.GraphKeys.GLOBAL_VARIABLES]
    if not isinstance(collections, (list, tuple, set)):
      raise ValueError(
          "collections argument to BloomFilterAdmitStrategy constructor must be "
          "a list, tuple, or set. Got % s type %s" % (collections, type(collections)))
    ops.add_to_collections(collections, self)

  def _calc_hash_func_num(self, false_positive_probability):
    log_fpp = abs(math.log(false_positive_probability, 2))
    return int(math.ceil(log_fpp))

  def _calc_bucket_size(self, max_element_size, false_positive_probability):
    log_fpp = abs(math.log(false_positive_probability))
    factor = math.log(2) ** 2
    bucket_size = int(math.ceil(log_fpp / factor * max_element_size))

    def is_prime(n):
      if n < 2:
        return False
      if n % 2 == 0 and n > 2:
        return False
      for i in range(3, int(math.sqrt(n) + 1), 2):
        if n % i == 0:
          return False
      return True

    while True:
      if is_prime(bucket_size):
        break
      bucket_size += 1
    return bucket_size

  def _optimal_dtype(self, minimum_frequency):
    if minimum_frequency < 2 ** 8:
      return dtypes.as_dtype('uint8')
    elif minimum_frequency < 2 ** 16:
      return dtypes.as_dtype('uint16')
    elif minimum_frequency < 2 ** 32:
      return dtypes.as_dtype('uint32')
    else:
      raise ValueError("Too large minimum_frequency is not supported.")

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
  def minimum_frequency(self):
    return self._minimum_frequency

  @property
  def max_element_size(self):
    return self._max_element_size

  @property
  def false_positive_probability(self):
    return self._false_positive_probability

  @property
  def slice_offset(self):
    return self._slice_offset

  @property
  def slice_size(self):
    return self._slice_size

  @property
  def max_slice_size(self):
    return self._max_slice_size

  @property
  def shape(self):
    return self._shape

  @property
  def dtype(self):
    return self._dtype

  @property
  def name(self):
    return self._name

  @property
  def distributed_name(self):
    return self._distributed_name

  @property
  def op(self):
    return self._handle.op

  @property
  def device(self):
    return self._handle.device

  @property
  def graph(self):
    return self._handle.graph

  def to_proto(v, export_scope=None):
    return None

  def from_proto(v, import_scope=None):
    return None

  def admit(self, keys, frequency):
    with ops.device(self.device):
      return gen_hash_ops.bloom_filter_admit_op(
          self._handle, keys, frequency)


@tf_export("hash_table.DistributedBloomFilterAdmitStrategy")
class DistributedBloomFilterAdmitStrategy(object):
  """TODO: Add DocString"""
  _DEFAULT_SLICE_SIZE = 65536
  def __init__(self,
               minimum_frequency,
               max_element_size=None,
               false_positive_probability=0.01,
               partitioner=None,
               slice_size=None,
               slicer=None,
               hash_tables=None,
               collections=None,
               name=None):
    if slice_size is None:
      slice_size = DistributedBloomFilterAdmitStrategy._DEFAULT_SLICE_SIZE

    if slicer is None and partitioner is None:
      raise ValueError("must specify one slice method or use slicer of hash_tables")

    if slicer is None:
      slicer = partitioner(slice_size)

    if hash_tables is None:
      hash_tables = []

    if max_element_size is None:
      max_element_size = BloomFilterAdmitStrategy._DEFAULT_ELEMENT_SIZE
    element_size_per_slice = (max_element_size + len(slicer) - 1) / len(slicer)
    strategies = []
    with ops.name_scope(name, "DistributedBloomFilter") as name:
      distributed_name = ops.name_from_scope_name(name)
      for i in range(len(slicer)):
        if slicer[i] >= slice_size:
          raise ValueError("slice overflow, offset=%s, slice_size=%s" % (slice[i], slice_size))
        hash_table = None if i >= len(hash_tables) else hash_tables[i]
        strategies.append(
          BloomFilterAdmitStrategy(minimum_frequency, element_size_per_slice,
            false_positive_probability, slicer[i], slice_size, hash_table,
            distributed_name, collections, "BloomFilter_" + str(i)))

    self._slice_size = slice_size
    self._slicer = slicer
    self._hash_tables = hash_tables
    self._name = distributed_name
    self._strategies = strategies

  @property
  def partitions(self):
    return list(self._strategies)

  @property
  def initializer(self):
    return [i.initializer for i in self._strategies]

  @property
  def device(self):
    return [i.device for i in self._strategies]

  @property
  def name(self):
    return self._name

  def admit(self, keys, frequency):
    flat_keys = array_ops.reshape(keys, [-1])
    flat_freq = array_ops.reshape(frequency, [-1])
    part_id = gen_hash_ops.hash_slice(self._slicer, flat_keys, self._slice_size)
    keys_per_device = data_flow_ops.dynamic_partition(flat_keys, part_id, len(self._slicer))
    freq_per_device = data_flow_ops.dynamic_partition(flat_freq, part_id, len(self._slicer))
    return [self._strategies[i].admit(keys_per_device[i], freq_per_device[i])
        for i in range(len(freq_per_device))]
