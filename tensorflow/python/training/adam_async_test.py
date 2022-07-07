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
"""Tests for AdamAsync."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import hash_table
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam_async


def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class AdamAsyncOptimizerTest(test.TestCase):

  def doTestSparse(self, use_resource=False):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      for use_gpu in [False, True]:
        with self.cached_session(use_gpu=use_gpu):
          # Initialize variables for numpy implementation.
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
          grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
          grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

          if use_resource:
            var0 = resource_variable_ops.ResourceVariable(var0_np)
            var1 = resource_variable_ops.ResourceVariable(var1_np)
          else:
            var0 = variables.Variable(var0_np)
            var1 = variables.Variable(var1_np)
          grads0_np_indices = np.array([0, 1], dtype=np.int32)
          grads0 = ops.IndexedSlices(
              constant_op.constant(grads0_np),
              constant_op.constant(grads0_np_indices), constant_op.constant([2]))
          grads1_np_indices = np.array([0, 1], dtype=np.int32)
          grads1 = ops.IndexedSlices(
              constant_op.constant(grads1_np),
              constant_op.constant(grads1_np_indices), constant_op.constant([2]))
          opt = adam_async.AdamAsyncOptimizer()
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          variables.global_variables_initializer().run()

          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], var0.eval())
          self.assertAllClose([3.0, 4.0], var1.eval())

          beta1_power = opt.get_slot(var0, 'beta1_power')
          beta2_power = opt.get_slot(var0, 'beta2_power')

          # Run 3 steps of AdamAsync
          for t in range(1, 4):
            self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
            self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
            update.run()

            var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, var0.eval())
            self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSparse(self):
    self.doTestSparse(use_resource=False)

  def testResourceSparse(self):
    self.doTestSparse(use_resource=True)

  def testSparseRepeatedIndices(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      for use_gpu in [False, True]:
        with self.cached_session(use_gpu=use_gpu):
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
          repeated_update = adam_async.AdamAsyncOptimizer().apply_gradients(
              [(grad_repeated_index, repeated_index_update_var)])
          aggregated_update = adam_async.AdamAsyncOptimizer().apply_gradients(
              [(grad_aggregated, aggregated_update_var)])
          variables.global_variables_initializer().run()
          self.assertAllClose(aggregated_update_var.eval(),
                              repeated_index_update_var.eval())
          for _ in range(3):
            repeated_update.run()
            aggregated_update.run()
            self.assertAllClose(aggregated_update_var.eval(),
                                repeated_index_update_var.eval())

  def doTestBasic(self, use_resource=False, use_callable_params=False):
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      for use_gpu in [True, False]:
        with self.session(graph=ops.Graph(), use_gpu=use_gpu):
          # Initialize variables for numpy implementation.
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
          grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
          grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

          if use_resource:
            var0 = resource_variable_ops.ResourceVariable(
                var0_np, name="var0_%d" % i)
            var1 = resource_variable_ops.ResourceVariable(
                var1_np, name="var1_%d" % i)
          else:
            var0 = variables.Variable(var0_np)
            var1 = variables.Variable(var1_np)
          grads0 = constant_op.constant(grads0_np)
          grads1 = constant_op.constant(grads1_np)

          learning_rate = lambda: 0.001
          beta1 = lambda: 0.9
          beta2 = lambda: 0.999
          epsilon = lambda: 1e-8
          if not use_callable_params:
            learning_rate = learning_rate()
            beta1 = beta1()
            beta2 = beta2()
            epsilon = epsilon()

          opt = adam_async.AdamAsyncOptimizer(learning_rate=learning_rate)
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          opt_variables = opt.variables()
          beta1_power = opt.get_slot(var0, 'beta1_power')
          beta2_power = opt.get_slot(var0, 'beta2_power')
          self.assertTrue(beta1_power is not None)
          self.assertTrue(beta2_power is not None)
          self.assertIn(beta1_power, opt_variables)
          self.assertIn(beta2_power, opt_variables)

          if not context.executing_eagerly():
            with ops.Graph().as_default():
              # Shouldn't return non-slot variables from other graphs.
              self.assertEqual(0, len(opt.variables()))
            self.evaluate(variables.global_variables_initializer())
            # Fetch params to validate initial values
            self.assertAllClose([1.0, 2.0], self.evaluate(var0))
            self.assertAllClose([3.0, 4.0], self.evaluate(var1))

          beta1_power = opt.get_slot(var0, 'beta1_power')
          beta2_power = opt.get_slot(var0, 'beta2_power')

          # Run 3 steps of AdamAsync
          for t in range(1, 4):
            if not context.executing_eagerly():
              self.evaluate(update)
            elif t > 1:
              opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            self.assertAllCloseAccordingToType(0.9**(t + 1),
                                              self.evaluate(beta1_power))
            self.assertAllCloseAccordingToType(0.999**(t + 1),
                                              self.evaluate(beta2_power))

            var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
            self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
            if use_resource:
              self.assertEqual("var0_%d/AdamAsync:0" % (i,),
                              opt.get_slot(var=var0, name="m").name)

  def testBasic(self):
    with self.cached_session():
      self.doTestBasic(use_resource=False)

  def testResourceBasic(self):
    self.doTestBasic(use_resource=True)

  def testTensorLearningRate(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      for use_gpu in [False, True]:
        with self.cached_session(use_gpu=use_gpu):
          # Initialize variables for numpy implementation.
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
          grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
          grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
          grads0 = constant_op.constant(grads0_np)
          grads1 = constant_op.constant(grads1_np)
          opt = adam_async.AdamAsyncOptimizer(constant_op.constant(0.001))
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          variables.global_variables_initializer().run()

          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], var0.eval())
          self.assertAllClose([3.0, 4.0], var1.eval())

          beta1_power = opt.get_slot(var0, 'beta1_power')
          beta2_power = opt.get_slot(var0, 'beta2_power')

          # Run 3 steps of AdamAsync
          for t in range(1, 4):
            self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
            self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
            update.run()

            var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, var0.eval())
            self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testSharing(self):
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      for use_gpu in [False, True]:
        with self.cached_session(use_gpu=use_gpu):
          # Initialize variables for numpy implementation.
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
          grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
          grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
          grads0 = constant_op.constant(grads0_np)
          grads1 = constant_op.constant(grads1_np)
          opt = adam_async.AdamAsyncOptimizer()
          update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          variables.global_variables_initializer().run()

          beta1_power = opt.get_slot(var0, 'beta1_power')
          beta2_power = opt.get_slot(var0, 'beta2_power')

          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], var0.eval())
          self.assertAllClose([3.0, 4.0], var1.eval())

          # Run 3 steps of intertwined AdamAsync1 and AdamAsync2.
          for t in range(1, 4):
            self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
            self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
            if t % 2 == 0:
              update1.run()
            else:
              update2.run()

            var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, var0.eval())
            self.assertAllCloseAccordingToType(var1_np, var1.eval())

  def testHashTable(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      for use_gpu in [False, True]:
        with self.cached_session(use_gpu=use_gpu) as sess:
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
          grads0_np = np.array([1.0, 1.0], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([1.0, 1.0], dtype=dtype.as_numpy_dtype)
          grads1_np = np.array([1.0, 1.0], dtype=dtype.as_numpy_dtype)

          with ops.name_scope("scope"):
            ht0 = hash_table.HashTable(
                [], dtype, "t1", init_ops.zeros_initializer)
            ht1 = hash_table.HashTable(
                [], dtype, "t2", init_ops.ones_initializer)
          res0 = ht0.lookup([0, 1])
          res1 = ht1.lookup([1, 2])
          loss = res0 + res1
          opt = adam_async.AdamAsyncOptimizer()
          step_op = opt.minimize(loss, var_list=[ht0, ht1])
          beta1_power = opt.get_slot(ht0, "beta1_power")
          beta2_power = opt.get_slot(ht0, "beta2_power")

          variables.global_variables_initializer().run()
          for t in range(1, 4):
            self.assertAllCloseAccordingToType(0.9**t, beta1_power.eval())
            self.assertAllCloseAccordingToType(0.999**t, beta2_power.eval())
            step_op.run()

            var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

            self.assertAllCloseAccordingToType(var0_np, res0.eval())
            self.assertAllCloseAccordingToType(var1_np, res1.eval())

if __name__ == "__main__":
  test.main()
