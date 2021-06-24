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
"""Tests for tensorflow.ops.trans_csv_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import googletest


class TransCsvID2SparseTest(test_util.TensorFlowTestCase):

  def testNormalProcess(self):
    x_1 = ["1,2", "3,4"]
    with self.test_session(use_gpu=False):
      y_tf = string_ops.trans_csv_id2sparse(x_1, max_id=10)
      expect_indices = [[0, 1], [0, 2], [1, 3], [1, 4]]
      expect_values = [1, 2, 3, 4]
      expect_dense_shape = [2, 10]
      re_i = y_tf.indices.eval() == expect_indices
      re_v = y_tf.values.eval() == expect_values
      re_d = y_tf.dense_shape.eval() == expect_dense_shape
      for line in re_i:
        for item in line:
          self.assertEqual(item, True)
      for item in re_v:
        self.assertEqual(item, True)
      for item in re_d:
        self.assertEqual(item, True)


class TransCsvKV2Dense(test_util.TensorFlowTestCase):

  def testNormalProcess(self):
    x_1 = ["1:0.1,2:0.2", "0:-.1e-1", "3:0.2,2:0.1"]
    with self.test_session(use_gpu=False) as sess:
      y_tf = string_ops.trans_csv_kv2dense(x_1, max_id=5)
      expect_values = [[0., 0.1, 0.2, 0., 0.],
                       [-0.01, 0., 0., 0., 0.],
                       [0., 0., 0.1, 0.2, 0.]]
      re_v = abs(y_tf.eval() - expect_values) < 0.0000001
      for line in re_v:
        for item in line:
          self.assertEqual(item, True)


class TransCsvToDense(test_util.TensorFlowTestCase):

  def testNormalProcess(self):
    x_1 = ["0.1,0.2 ", "-.1e-1", "0.2,  0.1, +.3"]
    with self.test_session(use_gpu=False) as sess:
      y_tf = string_ops.trans_csv_to_dense(x_1, max_id=3)
      expect_values = [[0.1, 0.2, 0.],
                       [-0.01, 0., 0.],
                       [0.2, 0.1, 0.3]]
      re_v = abs(y_tf.eval() - expect_values) < 0.0000001
      for line in re_v:
        for item in line:
          self.assertEqual(item, True)


if __name__ == "__main__":
  googletest.main()
