"""Tests for SparseApply of AdamOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import training_ops

class AdamSparseApplyTest(TensorFlowTestCase):

  def _toType(self, dtype):
    if dtype == np.float16:
      return dtypes.float16
    elif dtype == np.float32:
      return dtypes.float32
    elif dtype == np.float64:
      return dtypes.float64
    elif dtype == np.int32:
      return dtypes.int32
    elif dtype == np.int64:
      return dtypes.int64
    else:
      assert False, (dtype)

  def _testTypesForSparseAdam(self, x, m, v, t, lr, beta1, beta2, epsilon, grad, indices, use_gpu):
    self.setUp()
    with self.test_session(use_gpu=use_gpu):
      var = variables.VariableV1(x)
      m_a = variables.VariableV1(m)
      v_a = variables.VariableV1(v)
      variables.global_variables_initializer().run()
      beta1_power = beta1**t
      beta2_power = beta2**t

      self.assertAllCloseAccordingToType(x, var.eval())
      sparse_apply_adam = training_ops.sparse_apply_adam(
          var, m_a, v_a, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad,
          constant_op.constant(indices, self._toType(indices.dtype)))
      out = sparse_apply_adam.eval()
      self.assertShapeEqual(out, sparse_apply_adam)

      for (i, index) in enumerate(indices):
        new_var, new_m, new_v, = self._adamUpdateNumpy(x[index], grad[i], t, m[index], v[index], lr, beta1, beta2, epsilon)
        self.assertAllCloseAccordingToType(new_var, out[index])
        self.assertAllCloseAccordingToType(new_m, m_a.eval()[index])
        self.assertAllCloseAccordingToType(new_v, v_a.eval()[index])

  def testSparseApplyAdam(self):
    for (dtype, index_type, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [np.int32, np.int64], [False, True]):
      x_val = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
      m_val = [np.arange(1, 11), np.arange(11, 21), np.arange(21, 31)]
      v_val = [np.arange(2, 12), np.arange(12, 22), np.arange(22, 32)]
      x = np.array(x_val).astype(dtype)
      m = np.array(m_val).astype(dtype)
      v = np.array(v_val).astype(dtype)
      t = 1
      lr = np.array(1).astype(dtype)
      beta1 = np.array(2).astype(dtype)
      beta2 = np.array(3).astype(dtype)
      epsilon = np.array(4).astype(dtype)
      grad_val = [np.arange(10), np.arange(10)]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseAdam(x, m, v, t, lr, beta1, beta2, epsilon, grad, indices, use_gpu)

  def testSparseApplyAdamDim1(self):
    for (dtype, index_type, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [np.int32, np.int64], [False, True]):
      x_val = [[1.0], [2.0], [3.0]]
      m_val = [[4.0], [5.0], [6.0]]
      v_val = [[7.0], [8.0], [9.0]]
      x = np.array(x_val).astype(dtype)
      m = np.array(m_val).astype(dtype)
      v = np.array(v_val).astype(dtype)
      t = 1
      lr = np.array(1).astype(dtype)
      beta1 = np.array(2).astype(dtype)
      beta2 = np.array(3).astype(dtype)
      epsilon = np.array(4).astype(dtype)
      grad_val = [[1.5], [2.5]]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseAdam(x, m, v, t, lr, beta1, beta2, epsilon, grad, indices, use_gpu)

  def testApplyAdam(self):
    for dtype, use_gpu in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      var = np.arange(100).astype(dtype)
      m = np.arange(1, 101).astype(dtype)
      v = np.arange(101, 201).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForAdam(var, m, v, grad, use_gpu)

  def _testTypesForAdam(self, var, m, v, grad, use_gpu):
    self.setUp()
    with self.test_session(use_gpu=use_gpu):
      var_t = variables.VariableV1(var)
      m_t = variables.VariableV1(m)
      v_t = variables.VariableV1(v)

      t = 1
      beta1 = np.array(0.9, dtype=var.dtype)
      beta2 = np.array(0.999, dtype=var.dtype)
      beta1_power = beta1**t
      beta2_power = beta2**t
      lr = np.array(0.001, dtype=var.dtype)
      epsilon = np.array(1e-8, dtype=var.dtype)
      beta1_t = constant_op.constant(beta1, self._toType(var.dtype), [])
      beta2_t = constant_op.constant(beta2, self._toType(var.dtype), [])
      beta1_power_t = variables.VariableV1(beta1_power)
      beta2_power_t = variables.VariableV1(beta2_power)
      lr_t = constant_op.constant(lr, self._toType(var.dtype), [])
      epsilon_t = constant_op.constant(epsilon, self._toType(var.dtype), [])
      variables.global_variables_initializer().run()

      self.assertAllCloseAccordingToType(var, var_t.eval())
      new_var, _, _ = self._adamUpdateNumpy(var, grad, t, m, v, lr, beta1,
                                            beta2, epsilon)
      apply_adam = training_ops.apply_adam(var_t, m_t, v_t, beta1_power_t,
                                           beta2_power_t, lr_t, beta1_t,
                                           beta2_t, epsilon_t, grad)
      out = apply_adam.eval()
      self.assertShapeEqual(out, apply_adam)
      self.assertAllCloseAccordingToType(new_var, out)

  def _adamUpdateNumpy(self, param, g_t, t, m, v, alpha, beta1, beta2, epsilon):
    alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * g_t * g_t

    param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
    return param_t, m_t, v_t


if __name__ == '__main__':
  googletest.main()
