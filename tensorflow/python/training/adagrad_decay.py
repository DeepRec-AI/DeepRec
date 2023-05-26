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

"""Adagrad with accumulator decay for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import slot_creator
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.AdagradDecayOptimizer")
class AdagradDecayOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Adagrad algorithm with accumulator decay.
  Different from the original Adagrad algorithm, AdagradDecay performs decay 
  at given step with given rate. So that the accumulator will not be infinity.
  """

  def __init__(self, learning_rate, global_step, 
               initial_accumulator_value=0.1,
               accumulator_decay_step=100000, 
               accumulator_decay_rate=0.9,
               use_locking=False, name="AdagradDecay"):
    """Construct a new AdagradDecay optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      global_step: global step variable.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      accumulator_decay_step: When global_step reaches times of accumulator_decay_step,
        accumulator will be decayed with accumulator_decay_rate.
        accumulator *= accumulator_decay_rate
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "AdagradDecay".

    Raises:
      ValueError: If the `initial_accumulator_value`, `accumulator_decay_step` 
        or `accumulator_decay_rate` is invalid.
    """
    if initial_accumulator_value <= 0.0:
      raise ValueError("initial_accumulator_value must be positive: %s" %
                       initial_accumulator_value)
    if accumulator_decay_step <= 0:
      raise ValueError("accumulator_decay_step must be positive: %s" %
                       accumulator_decay_step)
    if accumulator_decay_rate <= 0.0 or accumulator_decay_rate >= 1.0:
      raise ValueError("accumulator_decay_rate must be in (0.0, 1.0): %s" %
                       accumulator_decay_rate)
    super(AdagradDecayOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._global_step = global_step
    self._initial_accumulator_value = initial_accumulator_value
    self._accumulator_decay_step = accumulator_decay_step
    self._accumulator_decay_rate = accumulator_decay_rate
    
    # Created in Initialize.
    self._learning_rate_tensor = None
    self._accumulator_decay_step_tensor = None
    self._accumulator_decay_rate_tensor = None
    self._accumulator_baseline_tensor = None
    self._global_step_on_worker = None

  def _create_slots(self, var_list):
    for v in var_list:
      with ops.colocate_with(v):
        dtype = v.dtype.base_dtype
        v_shape = v.get_shape()
        if v_shape.is_fully_defined():
          init = init_ops.constant_initializer(self._initial_accumulator_value,
                                               dtype=dtype)
        else:
          # Use a Tensor instead of initializer if variable does not have static
          # shape.
          init_constant = gen_array_ops.fill(array_ops.shape(v),
                                             self._initial_accumulator_value)
          init = math_ops.cast(init_constant, dtype)
      self._get_or_make_slot_with_initializer(v, init, v_shape, dtype,
                                              "accumulator", self._name,
                                              slot_config=slot_creator.SlotConfig(slot_index=1, slot_num=2))
      self._get_or_make_slot_with_initializer(
         v, init_ops.zeros_initializer(self._global_step.dtype), 
         v_shape, self._global_step.dtype, "accumulator_decay_power", self._name,
         slot_config=slot_creator.SlotConfig(slot_index=2, slot_num=2))
      # A slot to record how many times of decay has been operated on this index.
      # For a variable whose gradients are dense, only a scalar is needed. 
      # But we have not known that whose gradients are sparse until ApplyGradients.
      # So we create a slot with shape of the var's first dimension.
      # For case of sparse gradients, such as variable used for sparse embedding,
      # the slot will have the shape of the var's first dimension, which will be
      # updated based on the gradient indices.
      # decay_powers_shape = []
      # if v_shape.ndims is not None and v_shape.ndims > 0:
      #    decay_powers_shape.append(v_shape.dims[0])
      # self._get_or_make_slot_with_initializer(
      #    v, init_ops.zeros_initializer(self._global_step.dtype), 
      #    tensor_shape.TensorShape(decay_powers_shape), 
      #    self._global_step.dtype, "accumulator_decay_power", self._name)
      # The above code works correctly in UT, but makes some SHAPE error when a Saver used.
      # It occurs since TF1.4 and has not figured out the reason.
      # So we use a slot with the shape as v, which may introduce waste of memory.

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(
        self._learning_rate, name="learning_rate")
    self._accumulator_decay_step_tensor = math_ops.cast(
        ops.convert_to_tensor(self._accumulator_decay_step, 
                              name="accumulator_decay_step"),
        self._global_step.dtype.base_dtype)
    self._accumulator_decay_rate_tensor = ops.convert_to_tensor(
        self._accumulator_decay_rate, name="accumulator_decay_rate")
    self._accumulator_baseline_tensor = ops.convert_to_tensor(
        self._initial_accumulator_value, name="accumulator_baseline")
    # Performance optimization so that worker creates a copy of the global step
    # to avoid overloading the parameter server holding the global step.
    with ops.colocate_with(self._learning_rate_tensor):
      self._global_step_on_worker = array_ops.identity(self._global_step) + 1

  def _apply_dense(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    acc_decay_power = self.get_slot(var, "accumulator_decay_power")
    with ops.device(var.device):
      global_step = array_ops.identity(self._global_step_on_worker)
    return training_ops.apply_adagrad_decay(
        var,
        acc,
        acc_decay_power,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        self._accumulator_decay_step_tensor, 
        math_ops.cast(self._accumulator_decay_rate_tensor, var.dtype.base_dtype),
        math_ops.cast(self._accumulator_baseline_tensor, var.dtype.base_dtype),
        global_step,
        grad,
        use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    acc_decay_power = self.get_slot(var, "accumulator_decay_power")
    with ops.device(var.device):
      global_step = array_ops.identity(self._global_step_on_worker)
    return training_ops.resource_apply_adagrad_decay(
        var.handle,
        acc.handle,
        acc_decay_power.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        self._accumulator_decay_step_tensor,
        math_ops.cast(self._accumulator_decay_rate_tensor, grad.dtype.base_dtype),
        math_ops.cast(self._accumulator_baseline_tensor, grad.dtype.base_dtype),
        global_step,
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    acc_decay_power = self.get_slot(var, "accumulator_decay_power")
    with ops.device(var.device):
      global_step = array_ops.identity(self._global_step_on_worker)
    return training_ops.sparse_apply_adagrad_decay(
        var,
        acc,
        acc_decay_power,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        self._accumulator_decay_step_tensor, 
        math_ops.cast(self._accumulator_decay_rate_tensor, var.dtype.base_dtype),
        math_ops.cast(self._accumulator_baseline_tensor, var.dtype.base_dtype),
        global_step,
        grad.values,
        grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, indices_counts=None):
    acc = self.get_slot(var, "accumulator")
    acc_decay_power = self.get_slot(var, "accumulator_decay_power")
    with ops.device(var.device):
      global_step = array_ops.identity(self._global_step_on_worker)
    if isinstance(var, kv_variable_ops.EmbeddingVariable):
      if indices_counts != None:
        return training_ops.kv_resource_sparse_apply_adagrad_decay_with_counts(
          var.handle,
          acc.handle,
          acc_decay_power.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          self._accumulator_decay_step_tensor,
          math_ops.cast(self._accumulator_decay_rate_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._accumulator_baseline_tensor, grad.dtype.base_dtype),
          global_step,
          grad,
          indices,
          indices_counts,
          use_locking=self._use_locking)
      else:
        return training_ops.kv_resource_sparse_apply_adagrad_decay(
          var.handle,
          acc.handle,
          acc_decay_power.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          self._accumulator_decay_step_tensor,
          math_ops.cast(self._accumulator_decay_rate_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._accumulator_baseline_tensor, grad.dtype.base_dtype),
          global_step,
          grad,
          indices,
          use_locking=self._use_locking)
    else:
      return training_ops.resource_sparse_apply_adagrad_decay(
          var.handle,
          acc.handle,
          acc_decay_power.handle,
          math_ops.cast(self._learning_rate_tensor, grad.dtype),
          self._accumulator_decay_step_tensor,
          math_ops.cast(self._accumulator_decay_rate_tensor, grad.dtype.base_dtype),
          math_ops.cast(self._accumulator_baseline_tensor, grad.dtype.base_dtype),
          global_step,
          grad,
          indices,
          use_locking=self._use_locking)
