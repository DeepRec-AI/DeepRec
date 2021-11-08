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
"""Tests for SparseValidCutoff."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SparseSliceOpTest(test.TestCase):

  def _SparseTensor_4x6(self, val_dtype=np.int64):
    # [0 |  |2 |  |4 |5 ]
    # [  |11|  |13|14|  ]
    # [20|  |  |23|  |25]
    # [30|  |32|33|  |35]
    ind = np.array([[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1,
                                                                     4], [2, 0],
                    [2, 3], [2, 5], [3, 0], [3, 2], [3, 3], [3, 5]]).astype(
                        np.int64)
    val = np.array([0, 2, 4, 5, 11, 13, 14, 20, 23, 25, 30, 32, 33, 35]).astype(
        val_dtype)
    shape = np.array([4, 6]).astype(np.int64)
    return sparse_tensor.SparseTensor(ind, val, shape)

  def _SparseTensorValue_3x4x2(self):
    #  slice(:,:, 0)
    #  ['a0'|    |'b0'|    ]
    #  [    |'c0'|    |'d0']
    #  [    |    |'e0'|    ]
    #  slice(:,:, 1)
    #  ['a1'|    |'b1'|    ]
    #  [    |'c1'|    |'d1']
    #  [    |    |'e1'|    ]
    ind = np.array([[0, 0, 0], [0, 0, 1], [0, 2, 0], [0, 2, 1], [1, 1, 0],
                    [1, 1, 1], [1, 3, 0], [1, 3, 1], [2, 2, 0], [2, 2,
                                                                 1]]).astype(
                                                                     np.int64)
    val = np.array(['a0', 'a1', 'b0', 'b1', 'c0', 'c1', 'd0', 'd1', 'e0', 'e1'])
    shape = np.array([3, 4, 2]).astype(np.int64)
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def _SparseTensor_3x4x2(self):
    return sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_3x4x2())

  def testCutoff2D(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_4x6()
      sparse_tensor0 = sparse_ops.sparse_valid_cutoff(sp_input,
                                                      1,
                                                      4,
                                                      side="right")
      sparse_tensor1 = sparse_ops.sparse_valid_cutoff(sp_input,
                                                      1,
                                                      4,
                                                      side="left")
      sparse_tensor2 = sparse_ops.sparse_valid_cutoff(sp_input,
                                                      1,
                                                      8,
                                                      side="left")
      self.assertAllEqual(sparse_tensor0.indices.eval(),
                          [[0, 0], [0, 2], [1, 1], [1, 3], [2, 0], [2, 3], [3, 0], [3, 2], [3, 3]])
      self.assertAllEqual(sparse_tensor0.values.eval(), [0, 2, 11, 13, 20, 23, 30, 32, 33])
      self.assertAllEqual(sparse_tensor0.dense_shape.eval(), [4, 4])
      self.assertAllEqual(sparse_tensor1.indices.eval(),
                          [[0, 0], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 1], [2, 3], [3, 0], [3, 1], [3, 3]])
      self.assertAllEqual(sparse_tensor1.values.eval(), [2, 4, 5, 11, 13, 14, 23, 25, 32, 33, 35])
      self.assertAllEqual(sparse_tensor1.dense_shape.eval(), [4, 4])
      self.assertAllEqual(sparse_tensor2.indices.eval(),
                          [[0, 0], [0, 2], [0, 4], [0, 5],
                           [1, 1], [1, 3], [1, 4],
                           [2, 0], [2, 3], [2, 5],
                           [3, 0], [3, 2], [3, 3], [3, 5]])
      self.assertAllEqual(sparse_tensor2.values.eval(),
                          [0, 2, 4, 5,
                           11, 13, 14,
                           20, 23, 25,
                           30, 32, 33, 35])
      self.assertAllEqual(sparse_tensor2.dense_shape.eval(), [4, 8])

  def testCutoff3D(self):
    with self.test_session(use_gpu=False):
      sp_input = self._SparseTensor_3x4x2()
      sparse_tensor0 = sparse_ops.sparse_valid_cutoff(sp_input,
                                                      1,
                                                      2,
                                                      side="right")
      sparse_tensor1 = sparse_ops.sparse_valid_cutoff(sp_input,
                                                      1,
                                                      2,
                                                      side="left")
      sparse_tensor2 = sparse_ops.sparse_valid_cutoff(sp_input,
                                                      2,
                                                      1,
                                                      side="right")
      sparse_tensor3 = sparse_ops.sparse_valid_cutoff(sp_input,
                                                      1,
                                                      2,
                                                      side="right",
                                                      reverse=True)
      self.assertAllEqual(sparse_tensor0.indices.eval(),
                          [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]])
      self.assertAllEqual(sparse_tensor0.values.eval(), [b'a0', b'a1', b'c0', b'c1'])
      self.assertAllEqual(sparse_tensor0.dense_shape.eval(), [3, 2, 2])
      self.assertAllEqual(sparse_tensor1.indices.eval(),
                          [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1], [2, 1, 0], [2, 1, 1]])
      self.assertAllEqual(sparse_tensor1.values.eval(), [b'b0', b'b1', b'd0', b'd1', b'e0', b'e1'])
      self.assertAllEqual(sparse_tensor1.dense_shape.eval(), [3, 2, 2])
      self.assertAllEqual(sparse_tensor2.indices.eval(),
                          [[0, 0, 0], [0, 2, 0], [1, 1, 0], [1, 3, 0], [2, 2, 0]])
      self.assertAllEqual(sparse_tensor2.values.eval(), [b'a0', b'b0', b'c0', b'd0', b'e0'])
      self.assertAllEqual(sparse_tensor2.dense_shape.eval(), [3, 4, 1])
      self.assertAllEqual(sparse_tensor3.indices.eval(),
                          [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]])
      self.assertAllEqual(sparse_tensor3.values.eval(), [b'a0', b'a1', b'c0', b'c1'])
      self.assertAllEqual(sparse_tensor3.dense_shape.eval(), [3, 2, 2])


if __name__ == '__main__':
  test.main()
