# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Standard functions for creating slots.

A slot is a `Variable` created with the same first m-dimension as a primary
variable or `Tensor`. A slot is always scoped in the namespace of the primary
object and typically has the same device and type.

Slots are typically used as accumulators to track values associated with
the primary object:

```python
# Optimizers can create a slot for each variable to track accumulators
accumulators = {var : create_zeros_slot(var, "momentum") for var in vs}
for var in vs:
  apply_momentum(var, accumulators[var], lr, grad, momentum_tensor)

# Slots can also be used for moving averages
mavg = create_slot(var, var.initialized_value(), "exponential_moving_avg")
update_mavg = mavg.assign_sub((mavg - var) * (1 - decay))
```
"""
# pylint: disable=g-bad-name

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework.embedding import config_pb2
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import hash_table
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import distribution_strategy_context

class SlotConfig:
  def __init__(self,
               slot_num=1, slot_index=0,
               slot_type=config_pb2.SlotType.EMBEDDING_VARIABLE):
    self.slot_num = slot_num
    self.slot_index = slot_index
    self.slot_type = slot_type

def _is_embedding(v):
  """Returns true if v is something you get from a embedding variable."""
  return isinstance(v, kv_variable_ops.EmbeddingVariable)


def _create_slot_var(primary, val, scope, validate_shape, shape, dtype, slot_config):
  """Helper function for creating a slot variable."""

  # TODO(lukaszkaiser): Consider allowing partitioners to be set in the current
  # scope.
  current_partitioner = variable_scope.get_variable_scope().partitioner
  variable_scope.get_variable_scope().set_partitioner(None)
  # When init from val instead of callable initializer, the shape is expected to
  # be None, not <unknown> or any fully defined shape.
  shape = shape if callable(val) else None
  if resource_variable_ops.is_resource_variable(primary):
    use_resource = True
  elif isinstance(primary, variables.RefVariable):
    use_resource = False
  else:
    use_resource = None
  if isinstance(primary, kv_variable_ops.EmbeddingVariable):
    if slot_config is None:
      slot = variable_scope.get_embedding_variable_internal(
        scope,
        initializer=val,
        trainable=False,
        embedding_dim=shape,
        key_dtype=primary._invalid_key_type,
        validate_shape=validate_shape,
        steps_to_live=primary._steps_to_live,
        ht_partition_num=primary._ht_partition_num)
    else:
      filter_strategy = None
      if primary._filter_freq != 0:
        if primary._max_element_size != 0:
          filter_strategy = variables.CBFFilter(filter_freq=primary._filter_freq,
                                                max_element_size=primary._max_element_size,
                                                false_positive_probability=primary._false_positive_probability,
                                                counter_type=primary._counter_type)
        else:
          filter_strategy = variables.CounterFilter(filter_freq=primary._filter_freq)
      if slot_config.slot_type is config_pb2.SlotType.EMBEDDING_VARIABLE:
        primary.initializer._set_attr("slot_num", attr_value_pb2.AttrValue(i=slot_config.slot_num))
        primary._slot_num = slot_config.slot_num
        emb_index = primary._emb_index
        if primary.block_num > 1:
          primary = primary._primary
        slot = variable_scope.get_embedding_variable_v2_internal(
          scope, initializer=val, trainable=False,
          embedding_dim=shape, key_dtype=primary._invalid_key_type,
          validate_shape=validate_shape, 
          evconfig=variables.EmbeddingVariableConfig(
            steps_to_live=primary._steps_to_live,
            handle_name=primary._block_handle_name,
            emb_index=emb_index,
            block_num=primary.block_num,
            slot_index=slot_config.slot_index,
            primary=primary._primary,
            slot_num=slot_config.slot_num,
            storage_type=primary.storage_type,
            storage_path=primary._storage_path,
            storage_size=primary._storage_size,
            storage_cache_strategy=primary._storage_cache_strategy,
            l2_weight_threshold=primary._l2_weight_threshold,
            filter_strategy=filter_strategy)
        )
      else:
        slot = variable_scope.get_variable(
          scope,
          initializer=val,
          trainable=False,
          use_resource=use_resource,
          shape=shape,
          dtype=dtype,
          validate_shape=validate_shape)
  else:
    slot = variable_scope.get_variable(
        scope,
        initializer=val,
        trainable=False,
        use_resource=use_resource,
        shape=shape,
        dtype=dtype,
        validate_shape=validate_shape)
  variable_scope.get_variable_scope().set_partitioner(current_partitioner)

  # pylint: disable=protected-access
  if isinstance(primary, variables.Variable) and primary._save_slice_info:
    # Primary is a partitioned variable, so we need to also indicate that
    # the slot is a partitioned variable.  Slots have the same partitioning
    # as their primaries.
    # For examples when using AdamOptimizer in linear model, slot.name
    # here can be "linear//weights/Adam:0", while primary.op.name is
    # "linear//weight". We want to get 'Adam' as real_slot_name, so we
    # remove "'linear//weight' + '/'" and ':0'.
    real_slot_name = slot.name[len(primary.op.name + "/"):-2]
    slice_info = primary._save_slice_info
    if isinstance(slice_info, variables.Variable.SaveSliceInfo):
      # support slot's shape not same as primary's shape
      # example: primary's shape = [10, 20, 30], slot's shape =
      # None, [], [10], [10, 20] or [10, 20, 30] is allowed
      # slot's shape = None or [10, 20, 30], set slot's slice_info same as primary
      # slot's shape = [], don't set slot's slice_info
      # slot's shape = [10] or [10, 20], set slot's slice_info according to ndims
      n = slot.shape.ndims
      if n is None or n > 0:
        slot._set_save_slice_info(variables.Variable.SaveSliceInfo(
          slice_info.full_name + "/" + real_slot_name,
          slice_info.full_shape[:n],
          slice_info.var_offset[:n],
          slice_info.var_shape[:n],
          var_full_name=slice_info.var_full_name + "/" +
            real_slot_name if slice_info.var_full_name else None))
    else:
      slot._set_save_slice_info(
            slice_info.slot_save_slice_info(real_slot_name))
  # pylint: enable=protected-access
  return slot


def create_slot(primary, val, name, colocate_with_primary=True, slot_config=None):
  """Create a slot initialized to the given value.

  The type of the slot is determined by the given value.

  Args:
    primary: The primary `Variable` or `Tensor`.
    val: A `Tensor` specifying the initial value of the slot.
    name: Name to use for the slot variable.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.

  Returns:
    A `Variable` object.
  """
  # Scope the slot name in the namespace of the primary variable.
  # Set "primary.op.name + '/' + name" as default name, so the scope name of
  # optimizer can be shared when reuse is True. Meanwhile when reuse is False
  # and the same name has been previously used, the scope name will add '_N'
  # as suffix for unique identifications.
  validate_shape = val.get_shape().is_fully_defined()
  if context.executing_eagerly():
    prefix = primary._shared_name  # pylint: disable=protected-access
  else:
    prefix = primary.op.name
  with variable_scope.variable_scope(None, prefix + "/" + name):
    if colocate_with_primary:
      distribution_strategy = distribution_strategy_context.get_strategy()
      with distribution_strategy.extended.colocate_vars_with(primary):
        return _create_slot_var(primary, val, "", validate_shape, None, None, slot_config)
    else:
      return _create_slot_var(primary, val, "", validate_shape, None, None, slot_config)


def create_slot_with_initializer(primary, initializer, shape, dtype, name,
                                 colocate_with_primary=True, slot_config=None):
  """Creates a slot initialized using an `Initializer`.

  The type of the slot is determined by the given value.

  Args:
    primary: The primary `Variable` or `Tensor`.
    initializer: An `Initializer`.  The initial value of the slot.
    shape: Shape of the initial value of the slot.
    dtype: Type of the value of the slot.
    name: Name to use for the slot variable.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.

  Returns:
    A `Variable` object.
  """
  # Scope the slot name in the namespace of the primary variable.
  # Set "primary.op.name + '/' + name" as default name, so the scope name of
  # optimizer can be shared when reuse is True. Meanwhile when reuse is False
  # and the same name has been previously used, the scope name will add '_N'
  # as suffix for unique identifications.
  validate_shape = shape.is_fully_defined()
  if context.executing_eagerly():
    prefix = primary._shared_name  # pylint: disable=protected-access
  else:
    prefix = primary.op.name
  if isinstance(primary, hash_table.HashTable):
    with variable_scope.variable_scope(None, prefix + "/" + name):
      slot = primary.create_slot(shape,
                                 dtype,
                                 primary.distributed_name + "/slots/" + name,
                                 initializer,
                                 name=name)
      return slot
  with variable_scope.variable_scope(None, prefix + "/" + name):
    if colocate_with_primary:
      distribution_strategy = distribution_strategy_context.get_strategy()
      with distribution_strategy.extended.colocate_vars_with(primary):
        return _create_slot_var(primary, initializer, "", validate_shape, shape,
                                dtype, slot_config)
    else:
      return _create_slot_var(primary, initializer, "", validate_shape, shape,
                              dtype, slot_config)


def create_zeros_slot(primary, name, dtype=None, colocate_with_primary=True, slot_config=None):
  """Create a slot initialized to 0 with same shape as the primary object.

  Args:
    primary: The primary `Variable` or `Tensor`.
    name: Name to use for the slot variable.
    dtype: Type of the slot variable.  Defaults to the type of `primary`.
    colocate_with_primary: Boolean.  If True the slot is located
      on the same device as `primary`.

  Returns:
    A `Variable` object.
  """
  if dtype is None:
    dtype = primary.dtype
  slot_shape = primary.get_shape()
  if slot_shape.is_fully_defined():
    initializer = init_ops.zeros_initializer()
    return create_slot_with_initializer(
        primary, initializer, slot_shape, dtype, name,
        colocate_with_primary=colocate_with_primary,
        slot_config=slot_config)
  else:
    if isinstance(primary, variables.Variable):
      slot_shape = array_ops.shape(primary.initialized_value())
    else:
      slot_shape = array_ops.shape(primary)
    val = array_ops.zeros(slot_shape, dtype=dtype)
    return create_slot(primary, val, name,
                       colocate_with_primary=colocate_with_primary,
                       slot_config=slot_config)
