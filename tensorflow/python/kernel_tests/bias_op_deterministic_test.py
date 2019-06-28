# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for deterministic BiasAdd."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests import bias_op_base
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test

LayerShape = collections.namedtuple('LayerShape',
                                    'batch, height, width, depth, channels')

class BiasAddDeterministicTest(bias_op_base.BiasAddTestBase):

  def _random_data_op(self, shape):
    # np.random.random_sample can properly interpret either tf.TensorShape or
    # namedtuple as a list.
    return constant_op.constant(
        2 * np.random.random_sample(shape) - 1, dtype=dtypes.float32)

  def _random_ndarray(self, shape):
    return 2 * np.random.random_sample(shape) - 1

  def _assert_reproducible(self, operation):
    with self.cached_session(force_gpu=True):
      result_a = self.evaluate(operation)
      print("result_a: %r" % result_a)
      for i in range(1):
        result_b = self.evaluate(operation)
        print("result_b: %r" % result_a)
        self.assertAllEqual(result_a, result_b)

  # Working on getting this test to fail
  # Need to also iterate over rank, dtype, and data_format
  @test_util.run_cuda_only
  def testGradients(self):
    np.random.seed(3)
    channels = 8
    in_shape = LayerShape(batch=10, height=30, width=30, depth=30, channels=channels)
    bias_shape = (channels)
    out_shape = in_shape
    in_op = self._random_data_op(in_shape)
    bias_op = self._random_data_op(bias_shape)
    out_op = nn_ops.bias_add(in_op, bias_op, data_format="NHWC")
    upstream_gradients = variable_scope.get_variable("upstream_gradients", out_shape)
    with self.cached_session(force_gpu=True):
      self.evaluate(upstream_gradients.initializer)
    for i in range(2):
      # I tried using a variable for grad_ys and assiging new values to it, but
      # it seems that the gradients function captures the values at initialization
      # and never updates them. So I ended up re-instantiating the gradient op
      # on each iteration, which is very slow and inefficient.
      # upstream_gradients.assign(self._random_ndarray(out_shape))
      bias_gradients_op = gradients_impl.gradients(out_op, bias_op,
                                                   grad_ys=self._random_data_op(out_shape),
                                                   colocate_gradients_with_ops=True)
      self._assert_reproducible(bias_gradients_op)

  def testInputDims(self):
    pass

  def testBiasVec(self):
    pass

  def testBiasInputsMatch(self):
    pass


if __name__ == "__main__":
  os.environ["TF_DETERMINISTIC_OPS"] = "1"
  test.main()
