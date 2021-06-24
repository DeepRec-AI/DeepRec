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
"""Tests for tensorflow.ops.decode_[dense, sparse]/decode_kv2[dense, sparse]."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import googletest
from tensorflow.core.example import feature_pb2


class DecodeDenseTest(test_util.TensorFlowTestCase):

  def testFloatProcess(self):
    with self.test_session(use_gpu=False):
      float_list=feature_pb2.FloatList(value=[2.1, 3.2, 4.3])
      y = string_ops.decode_dense([float_list.SerializeToString()])
      y_ = y.eval()
      self.assertAlmostEqual(2.1, y_[0], delta=1e-05)
      self.assertAlmostEqual(3.2, y_[1], delta=1e-05)
      self.assertAlmostEqual(4.3, y_[2], delta=1e-05)

  def testInt32Process(self):
    with self.test_session(use_gpu=False):
      int32_list=feature_pb2.Int32List(value=[2, 3, 4])
      y = string_ops.decode_dense_int32([int32_list.SerializeToString()])
      y_ = y.eval()
      self.assertAllEqual(y_, [2, 3, 4])

class DecodeSparseTest(test_util.TensorFlowTestCase):

  def testNormalrocess(self):
    with self.test_session(use_gpu=False):
      int32_list1=feature_pb2.Int32List(value=[2, 3])
      int32_list2=feature_pb2.Int32List(value=[4])
      y = string_ops.decode_sparse([int32_list1.SerializeToString(),
                                    int32_list2.SerializeToString()], max_id=100)
      indices_ = y.indices.eval()
      values_ = y.values.eval()
      dense_shape_ = y.dense_shape.eval()
      self.assertAllEqual(indices_, [[0, 2], [0, 3], [1, 4]])
      self.assertAllEqual(values_, [2, 3, 4])
      self.assertAllEqual(dense_shape_, [2, 100])

class DecodeKV2DenseTest(test_util.TensorFlowTestCase):

  def testNormalrocess(self):
    with self.test_session(use_gpu=False):
      kv_list1=feature_pb2.KvList(id=[2, 3],value=[2.1, 3.2])
      kv_list2=feature_pb2.KvList(id=[4],value=[4.3])
      y = string_ops.decode_kv2dense([kv_list1.SerializeToString(),
                                      kv_list2.SerializeToString()], max_col=5)
      y_ = y[0].eval()
      self.assertAlmostEqual(2.1, y_[2], delta=1e-05)
      self.assertAlmostEqual(3.2, y_[3], delta=1e-05)
      self.assertAlmostEqual(4.3, y_[4+5], delta=1e-05)

class DecodeKV2SparseTest(test_util.TensorFlowTestCase):

  def testNormalrocess(self):
    with self.test_session(use_gpu=False):
      kv_list1=feature_pb2.KvList(id=[2, 3],value=[2.1, 3.2])
      kv_list2=feature_pb2.KvList(id=[4],value=[4.3])
      y = string_ops.decode_kv2sparse([kv_list1.SerializeToString(),
                                       kv_list2.SerializeToString()], max_id=100)
      indices_ = y.indices.eval()
      values_ = y.values.eval()
      dense_shape_ = y.dense_shape.eval()
      self.assertAllEqual(indices_, [[0, 2], [0, 3], [1, 4]])
      self.assertAlmostEqual(2.1, values_[0], delta=1e-05)
      self.assertAlmostEqual(3.2, values_[1], delta=1e-05)
      self.assertAlmostEqual(4.3, values_[2], delta=1e-05)
      self.assertAllEqual(dense_shape_, [2, 100])

if __name__ == "__main__":
  googletest.main()
