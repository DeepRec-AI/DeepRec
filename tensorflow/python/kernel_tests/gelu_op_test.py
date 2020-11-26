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
"""Tests for Gelu and GeluGrad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class GeluTest(test.TestCase):

  def _npGelu(self, x, approximate=False):
    if approximate:
      return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) *
                                      (x + 0.044715 * np.power(x, 3))))
    else:
      from scipy.stats import norm  # pylint: disable=g-import-not-at-top
      return x * norm.cdf(x)

  def _testGelu(self, np_features, approximate=False, use_gpu=False):
    expected_values = self._npGelu(np_features, approximate)
    with self.cached_session(use_gpu=use_gpu):
      results = self.evaluate(nn_ops.gelu(np_features, approximate))
    self.assertAllCloseAccordingToType(expected_values, results)

  def testNumbers(self):
    for approximate in [False, True]:
      for t in [np.float16, np.float32, np.float64]:
        self._testGelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            approximate, use_gpu=False)
        self._testGelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            approximate, use_gpu=True)

  def testGradient(self):
    with self.cached_session():
      for approximate in [False, True]:
        x = constant_op.constant(
            [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
            shape=[2, 5])
        theoretical, numerical = gradient_checker_v2.compute_gradient(
            lambda a: nn_ops.gelu(a, approximate), [x])
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(
            lambda a: nn_ops.gelu(a, approximate), [x]))

    print("gelu (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

if __name__ == "__main__":
  test.main()
