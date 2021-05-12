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
"""Functional tests for aggregate operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad_decay_v2


class AdagradDecayOptimizerV2Test(test.TestCase):

  def doTestBasic(self, use_locking=False, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      for global_step_dtype in [dtypes.int32, dtypes.int64]:
        global_step = variables.Variable(0, dtype=global_step_dtype)
        global_step_update = state_ops.assign_add(global_step, 1)
        with self.test_session():
          if use_resource:
            var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
            var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
          else:
            var0 = variables.Variable([1.0, 2.0], dtype=dtype)
            var1 = variables.Variable([3.0, 4.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
          grads1 = constant_op.constant([0.2, 0.2], dtype=dtype)
          ada_decay_opt = adagrad_decay_v2.AdagradDecayOptimizerV2(
              1.0, global_step, accumulator_decay_step=3, accumulator_decay_rate=0.9, 
              initial_accumulator_value=0.1, use_locking=use_locking)
          ada_decay_update = ada_decay_opt.apply_gradients(
              zip([grads0, grads1], [var0, var1]))
          variables.global_variables_initializer().run()
          variables.local_variables_initializer().run()
        
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], var0.eval())
          self.assertAllClose([3.0, 4.0], var1.eval())
        
          # Run 3 steps of adagraddecay
          v0_expect = 1.0
          v1_expect = 3.0
          v0_accum = 0.1
          v1_accum = 0.1
          for _ in range(4):
            ada_decay_update.run()
            global_step_update.eval()
            if _ == 2:
              v0_accum = v0_accum * 0.9      # accum *= decay_rate 
              v1_accum = v1_accum * 0.9
            v0_accum = v0_accum + 0.1 * 0.1  # accum += g^2
            v1_accum = v1_accum + 0.2 * 0.2  # accum += g^2
            v0_expect = v0_expect - 1.0 / np.sqrt(v0_accum) * 0.1   # v -= learning_rate * / sqrt(accum) * g
            v1_expect = v1_expect - 1.0 / np.sqrt(v1_accum) * 0.2   # v -= learning_rate * / sqrt(accum) * g
        
          # Validate updated params
          self.assertAllCloseAccordingToType(
              np.array([v0_expect, v0_expect + 1]), var0.eval())
          self.assertAllCloseAccordingToType(
              np.array([v1_expect, v1_expect + 1]), var1.eval())

  def testBasic(self):
    self.doTestBasic(use_locking=False)

  def testBasicResource(self):
    self.doTestBasic(use_locking=False, use_resource=True)

  def testBasicLocked(self):
    self.doTestBasic(use_locking=True)

  def testBasicResourceLocked(self):
    self.doTestBasic(use_locking=True, use_resource=True)

  def doTestSparseBasic(self, use_resource):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      for global_step_dtype in [dtypes.int32, dtypes.int64]:
        global_step = variables.Variable(0, dtype=global_step_dtype)
        global_step_update = state_ops.assign_add(global_step, 1)
        with self.test_session():
          if use_resource:
            var0 = resource_variable_ops.ResourceVariable([[1.0], [2.0]], dtype=dtype)
            var1 = resource_variable_ops.ResourceVariable([[3.0], [4.0]], dtype=dtype)
          else:
            var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
            var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
          grads0 = ops.IndexedSlices(
              constant_op.constant(
                  [0.1], shape=[1, 1], dtype=dtype),
              constant_op.constant([0]),
              constant_op.constant([2, 1]))
          grads1 = ops.IndexedSlices(
              constant_op.constant(
                  [0.2], shape=[1, 1], dtype=dtype),
              constant_op.constant([1]),
              constant_op.constant([2, 1]))
          ada_decay_opt = adagrad_decay_v2.AdagradDecayOptimizerV2(
              1.0, global_step, initial_accumulator_value=0.1, 
              accumulator_decay_step=3, accumulator_decay_rate=0.9) 
          ada_decay_update = ada_decay_opt.apply_gradients(
              zip([grads0, grads1], [var0, var1]))
          accummm = ada_decay_opt.get_slot(var0, "accumulator")
          variables.local_variables_initializer().run()
          variables.global_variables_initializer().run()
        
          # Fetch params to validate initial values
          self.assertAllClose([[1.0], [2.0]], var0.eval())
          self.assertAllClose([[3.0], [4.0]], var1.eval())
        
          # Run 3 steps of adagraddecay
          v0_expect = 1.0
          v1_expect = 3.0
          v0_accum = 0.1
          v1_accum = 0.1
          for _ in range(4):
            ada_decay_update.run()
            global_step_update.eval()
            if _ == 2:
              v0_accum = v0_accum * 0.9      # accum *= decay_rate 
              v1_accum = v1_accum * 0.9
            v0_accum = v0_accum + 0.1 * 0.1  # accum += g^2
            v1_accum = v1_accum + 0.2 * 0.2  # accum += g^2
            v0_expect = v0_expect - 1.0 / np.sqrt(v0_accum) * 0.1   # v -= learning_rate * / sqrt(accum) * g
            v1_expect = v1_expect - 1.0 / np.sqrt(v1_accum) * 0.2   # v -= learning_rate * / sqrt(accum) * g
        
          # Validate updated params
          self.assertAllCloseAccordingToType(
              np.array([[v0_expect], [2.0]]), var0.eval())
          self.assertAllCloseAccordingToType(
              np.array([[3.0], [v1_expect + 1]]), var1.eval())

  def testSparseBasic(self):
    self.doTestSparseBasic(False)

  def testSparseBasicResource(self):
    self.doTestSparseBasic(True)
  
  def testSparseRepeatedIndices(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      global_step1 = variables.Variable(0, dtype=dtypes.int64)
      global_step2 = variables.Variable(0, dtype=dtypes.int64)
      with self.test_session():
        repeated_index_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        aggregated_update_var = variables.Variable(
            [[1.0], [2.0]], dtype=dtype)
        grad_repeated_index = ops.IndexedSlices(
            constant_op.constant(
                [0.1, 0.1], shape=[2, 1], dtype=dtype),
            constant_op.constant([1, 1]),
            constant_op.constant([2, 1]))
        grad_aggregated = ops.IndexedSlices(
            constant_op.constant(
                [0.2], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]))
        opt1 = adagrad_decay_v2.AdagradDecayOptimizerV2(3.0, global_step1, accumulator_decay_step=2)
        opt2 = adagrad_decay_v2.AdagradDecayOptimizerV2(3.0, global_step2, accumulator_decay_step=2)
        repeated_update = opt1.apply_gradients([(grad_repeated_index, repeated_index_update_var)])
        aggregated_update = opt2.apply_gradients([(grad_aggregated, aggregated_update_var)])
        global_step1_update = state_ops.assign_add(global_step1, 1)
        global_step2_update = state_ops.assign_add(global_step2, 1)
        variables.global_variables_initializer().run()
        self.assertAllClose(aggregated_update_var.eval(),
                            repeated_index_update_var.eval())
        for _ in range(3):
          repeated_update.run()
          aggregated_update.run()
          global_step1_update.eval()
          global_step2_update.eval()
          self.assertAllClose(aggregated_update_var.eval(),
                              repeated_index_update_var.eval())

  def doTestBaseline(self, use_locking=False, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      for global_step_dtype in [dtypes.int32, dtypes.int64]:
        decay_rate = 0.6
        decay_step = 3
        init_accum = 1.0
        global_step = variables.Variable(0, dtype=global_step_dtype)
        global_step_update = state_ops.assign_add(global_step, 1)
        with self.test_session():
          if use_resource:
            var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
            var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
          else:
            var0 = variables.Variable([1.0, 2.0], dtype=dtype)
            var1 = variables.Variable([3.0, 4.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
          grads1 = constant_op.constant([0.2, 0.2], dtype=dtype)
          ada_decay_opt = adagrad_decay_v2.AdagradDecayOptimizerV2(
              1.0, global_step, accumulator_decay_step=decay_step, accumulator_decay_rate=decay_rate, 
              initial_accumulator_value=init_accum, use_locking=use_locking)
          ada_decay_update = ada_decay_opt.apply_gradients(
              zip([grads0, grads1], [var0, var1]))

          accum0 = ada_decay_opt.get_slot(var0, 'accumulator')
          accum1 = ada_decay_opt.get_slot(var1, 'accumulator')
          variables.global_variables_initializer().run()
          variables.local_variables_initializer().run()
        
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 1.0], accum0.eval())
          self.assertAllClose([1.0, 1.0], accum1.eval())
        
          # Run 3 steps of adagraddecay
          v0_accum = init_accum
          v1_accum = init_accum
          for _ in range(4):
            ada_decay_update.run()
            global_step_update.eval()
            if _ == 2:
              v0_accum = v0_accum * decay_rate      # accum *= decay_rate 
              v1_accum = v1_accum * decay_rate
              if v0_accum < init_accum:
                v0_accum = init_accum
              if v1_accum < init_accum:
                v1_accum = init_accum
            v0_accum = v0_accum + 0.1 * 0.1  # accum += g^2
            v1_accum = v1_accum + 0.2 * 0.2  # accum += g^2
        
          # Validate updated params
          self.assertAllCloseAccordingToType(
              np.array([v0_accum, v0_accum]), accum0.eval())
          self.assertAllCloseAccordingToType(
              np.array([v1_accum, v1_accum]), accum1.eval())
  
  def testBaseLine(self):
    self.doTestBaseline(use_locking=False)

  def testBaselineResource(self):
    self.doTestBaseline(use_locking=False, use_resource=True)

  def testBaselineLocked(self):
    self.doTestBaseline(use_locking=True)

  def testBaselineResourceLocked(self):
    self.doTestBaseline(use_locking=True, use_resource=True)

  def doTestSparseBaseline(self, use_resource):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      for global_step_dtype in [dtypes.int32, dtypes.int64]:
        global_step = variables.Variable(0, dtype=global_step_dtype)
        global_step_update = state_ops.assign_add(global_step, 1)
        init_accum = 1.0
        with self.test_session():
          if use_resource:
            var0 = resource_variable_ops.ResourceVariable([[1.0], [2.0]], dtype=dtype)
            var1 = resource_variable_ops.ResourceVariable([[3.0], [4.0]], dtype=dtype)
          else:
            var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
            var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
          grads0 = ops.IndexedSlices(
              constant_op.constant(
                  [0.1], shape=[1, 1], dtype=dtype),
              constant_op.constant([0]),
              constant_op.constant([2, 1]))
          grads1 = ops.IndexedSlices(
              constant_op.constant(
                  [0.2], shape=[1, 1], dtype=dtype),
              constant_op.constant([1]),
              constant_op.constant([2, 1]))
          ada_decay_opt = adagrad_decay_v2.AdagradDecayOptimizerV2(
              1.0, global_step, initial_accumulator_value=init_accum, 
              accumulator_decay_step=3, accumulator_decay_rate=0.6) 
          ada_decay_update = ada_decay_opt.apply_gradients(
              zip([grads0, grads1], [var0, var1]))
          accum0 = ada_decay_opt.get_slot(var0, "accumulator")
          accum1 = ada_decay_opt.get_slot(var1, "accumulator")
          variables.local_variables_initializer().run()
          variables.global_variables_initializer().run()
        
          # Fetch params to validate initial values
          self.assertAllClose([[1.0], [1.0]], accum0.eval())
          self.assertAllClose([[1.0], [1.0]], accum1.eval())
        
          # Run 3 steps of adagraddecay
          v0_accum = init_accum
          v1_accum = init_accum
          for _ in range(4):
            ada_decay_update.run()
            global_step_update.eval()
            if _ == 2:
              v0_accum = v0_accum * 0.6      # accum *= decay_rate 
              v1_accum = v1_accum * 0.6
              if v0_accum < init_accum:
                v0_accum = init_accum
              if v1_accum < init_accum:
                v1_accum = init_accum
            v0_accum = v0_accum + 0.1 * 0.1  # accum += g^2
            v1_accum = v1_accum + 0.2 * 0.2  # accum += g^2
        
          # Validate updated params
          self.assertAllCloseAccordingToType(
              np.array([[v0_accum], [1.0]]), accum0.eval())
          self.assertAllCloseAccordingToType(
              np.array([[1.0], [v1_accum]]), accum1.eval())

  def testSparseBaseline(self):
    self.doTestSparseBaseline(False)

  def testSparseBaselineResource(self):
    self.doTestSparseBaseline(True)
  
if __name__ == "__main__":
  test.main()
