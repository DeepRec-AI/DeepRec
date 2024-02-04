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
"""Tests for tensorflow.kernels.unique_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

# set environ before tf initializing global varialbes
PreservedKey = 1 << 10
os.environ["DEEPREC_CONFIG_RAND_64"] = str(PreservedKey)

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test


class UniqueTest(test.TestCase):

  def testInt32(self):
    x = np.random.randint(0, high=1000, size=700000)
    with self.cached_session(use_gpu=True) as sess:
      y, idx = array_ops.unique(x)
      tf_y, tf_idx = sess.run([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testInt32OutIdxInt64(self):
    x = np.random.randint(2, high=1000, size=700000)
    with self.cached_session(use_gpu=True) as sess:
      y, idx = array_ops.unique(x, out_idx=dtypes.int64)
      tf_y, tf_idx = sess.run([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testInt64OutIdxInt64(self):
    np.random.seed(0)
    x = np.random.randint(-1000000000, high=1000000000, size=1000000,
      dtype=np.int64)
    with self.cached_session(use_gpu=True) as sess:
      y, idx = array_ops.unique(x, out_idx=dtypes.int64)
      tf_y, tf_idx = sess.run([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testInt64OutIdxInt32(self):
    np.random.seed(0)
    x = np.random.randint(-1000000000, high=1000000000, size=1000000,
      dtype=np.int64)
    with self.cached_session(use_gpu=True) as sess:
      y, idx = array_ops.unique(x, out_idx=dtypes.int32)
      tf_y, tf_idx = sess.run([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testString(self):
    indx = np.random.randint(65, high=122, size=70000)
    x = [chr(i) for i in indx]
    with self.cached_session() as sess:
      y, idx = array_ops.unique(x)
      tf_y, tf_idx = sess.run([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))

  def testInt32Axis(self):
    for dtype in [np.int32, np.int64]:
      x = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]])
      with self.cached_session() as sess:
        y0, idx0 = gen_array_ops.unique_v2(x, axis=np.array([0], dtype))
        tf_y0, tf_idx0 = sess.run([y0, idx0])
        y1, idx1 = gen_array_ops.unique_v2(x, axis=np.array([1], dtype))
        tf_y1, tf_idx1 = sess.run([y1, idx1])
      self.assertAllEqual(tf_y0, np.array([[1, 0, 0], [2, 0, 0]]))
      self.assertAllEqual(tf_idx0, np.array([0, 0, 1]))
      self.assertAllEqual(tf_y1, np.array([[1, 0], [1, 0], [2, 0]]))
      self.assertAllEqual(tf_idx1, np.array([0, 1, 1]))

  def testInt32V2(self):
    # This test is only temporary, once V2 is used
    # by default, the axis will be wrapped to allow `axis=None`.
    x = np.random.randint(2, high=10, size=7000)
    with self.cached_session() as sess:
      y, idx = gen_array_ops.unique_v2(x, axis=np.array([], np.int32))
      tf_y, tf_idx = sess.run([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def IllegalIdForMultMapUnique(self):
    recover_env = False
    if 'DEEPREC_UNIQUE_OP_PARTITION_SIZE' in os.environ:
      recover_env = True
      old_env = os.environ['DEEPREC_UNIQUE_OP_PARTITION_SIZE']
    os.environ['DEEPREC_UNIQUE_OP_PARTITION_SIZE'] = '2'

    with self.cached_session() as sess:
      x = np.array([-1, 0, 1, PreservedKey], dtype=np.int64)
      y, idx = array_ops.unique(x, out_idx=dtypes.int64)
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Input id is preserved key of dense_hash_map, "
          "not supported: " + str(PreservedKey)):
        tf_y, tf_idx = sess.run([y, idx])

    del os.environ['DEEPREC_UNIQUE_OP_PARTITION_SIZE']
    if recover_env:
      os.environ['DEEPREC_UNIQUE_OP_PARTITION_SIZE'] = old_env

  def RunUniqueWithDifferentMaps(self, map_type,
                                 test_illegal_key=False):
    recover_env = False
    if 'DEEPREC_UNIQUE_OP_HASH_MAP' in os.environ:
      recover_env = True
      old_env = os.environ['DEEPREC_UNIQUE_OP_HASH_MAP']

    os.environ['DEEPREC_UNIQUE_OP_HASH_MAP'] = map_type
    self.testInt32()
    self.testInt32OutIdxInt64()
    self.testInt64OutIdxInt64()
    self.testInt64OutIdxInt32()
    self.testInt32Axis()
    self.testInt32V2()
    if test_illegal_key:
      self.IllegalIdForMultMapUnique()

    del os.environ['DEEPREC_UNIQUE_OP_HASH_MAP']
    if recover_env:
      os.environ['DEEPREC_UNIQUE_OP_HASH_MAP'] = old_env

  def testUniqueMultiMap(self):
    self.RunUniqueWithDifferentMaps('MULTIMAP', True)

  def testUniqueStlMap(self):
    self.RunUniqueWithDifferentMaps('STL')

  def testUniqueAbslMap(self):
    self.RunUniqueWithDifferentMaps('ABSL')

  def testUniqueDenseHashMap(self):
    self.RunUniqueWithDifferentMaps('GOOGLE')

class UniqueWithCountsTest(test.TestCase):

  def testInt32(self):
    x = np.random.randint(2, high=1000, size=700000)
    with self.cached_session() as sess:
      y, idx, count = array_ops.unique_with_counts(x)
      tf_y, tf_idx, tf_count = sess.run([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testInt32OutIdxInt64(self):
    x = np.random.randint(2, high=1000, size=700000)
    with self.cached_session() as sess:
      y, idx, count = array_ops.unique_with_counts(x, out_idx=dtypes.int64)
      tf_y, tf_idx, tf_count = sess.run([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testString(self):
    indx = np.random.randint(65, high=122, size=7000)
    x = [chr(i) for i in indx]

    with self.cached_session() as sess:
      y, idx, count = array_ops.unique_with_counts(x)
      tf_y, tf_idx, tf_count = sess.run([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))
    for value, count in zip(tf_y, tf_count):
      v = [1 if x[i] == value.decode('ascii') else 0 for i in range(7000)]
      self.assertEqual(count, sum(v))

  def testInt32Axis(self):
    for dtype in [np.int32, np.int64]:
      x = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]])
      with self.cached_session() as sess:
        y0, idx0, count0 = gen_array_ops.unique_with_counts_v2(
            x, axis=np.array([0], dtype))
        tf_y0, tf_idx0, tf_count0 = sess.run([y0, idx0, count0])
        y1, idx1, count1 = gen_array_ops.unique_with_counts_v2(
            x, axis=np.array([1], dtype))
        tf_y1, tf_idx1, tf_count1 = sess.run([y1, idx1, count1])
      self.assertAllEqual(tf_y0, np.array([[1, 0, 0], [2, 0, 0]]))
      self.assertAllEqual(tf_idx0, np.array([0, 0, 1]))
      self.assertAllEqual(tf_count0, np.array([2, 1]))
      self.assertAllEqual(tf_y1, np.array([[1, 0], [1, 0], [2, 0]]))
      self.assertAllEqual(tf_idx1, np.array([0, 1, 1]))
      self.assertAllEqual(tf_count1, np.array([1, 2]))

  def testInt32V2(self):
    # This test is only temporary, once V2 is used
    # by default, the axis will be wrapped to allow `axis=None`.
    x = np.random.randint(2, high=10, size=7000)
    with self.cached_session() as sess:
      y, idx, count = gen_array_ops.unique_with_counts_v2(
          x, axis=np.array([], np.int32))
      tf_y, tf_idx, tf_count = sess.run([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def RunUniqueWithCountsWithDifferentMaps(self, map_type):
    recover_env = False
    if 'DEEPREC_UNIQUE_OP_HASH_MAP' in os.environ:
      recover_env = True
      old_env = os.environ['DEEPREC_UNIQUE_OP_HASH_MAP']

    os.environ['DEEPREC_UNIQUE_OP_HASH_MAP'] = map_type
    self.testInt32()
    self.testInt32OutIdxInt64()
    self.testInt32Axis()
    self.testInt32V2()

    del os.environ['DEEPREC_UNIQUE_OP_HASH_MAP']
    if recover_env:
      os.environ['DEEPREC_UNIQUE_OP_HASH_MAP'] = old_env

  def testUniqueWithCountsMultiMap(self):
    self.RunUniqueWithCountsWithDifferentMaps('MULTIMAP')

  def testUniqueWithCountsStlMap(self):
    self.RunUniqueWithCountsWithDifferentMaps('STL')

  def testUniqueWithCountsAbslMap(self):
    self.RunUniqueWithCountsWithDifferentMaps('ABSL')

  def testUniqueWithCountsDenseHashMap(self):
    self.RunUniqueWithCountsWithDifferentMaps('GOOGLE')

class UniqueWithExtraCountsTest(test.TestCase):

  def testInt32(self):
    x = np.random.randint(2, high=1000, size=700000)
    extra_x = x[:5].tolist()
    extra_x_tensor = [constant_op.constant(extra_x, dtypes.int64)]
    extra_count = [500 for _ in range(5)]
    extra_count_tensor = [constant_op.constant(extra_count, dtypes.int32)]
    with self.cached_session() as sess:
      y, idx, count = gen_array_ops._unique_with_extra_counts(x, extra_x_tensor, extra_count_tensor)
      tf_y, tf_idx, tf_count = sess.run([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      if value in extra_x:
        self.assertEqual(count, np.sum(x == value) + 499)
      else:
        self.assertEqual(count, np.sum(x == value))

  def testInt32OutIdxInt64(self):
    x = np.random.randint(2, high=1000, size=700000)
    extra_x = x[:5].tolist()
    extra_x_tensor = [constant_op.constant(extra_x, dtypes.int64)]
    extra_count = [500 for _ in range(5)]
    extra_count_tensor = [constant_op.constant(extra_count, dtypes.int64)]
    with self.cached_session() as sess:
      y, idx, count = gen_array_ops._unique_with_extra_counts(x, extra_x_tensor, extra_count_tensor)
      tf_y, tf_idx, tf_count = sess.run([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      if value in extra_x:
        self.assertEqual(count, np.sum(x == value) + 499)
      else:
        self.assertEqual(count, np.sum(x == value))

  def RunUniqueWithCountsWithDifferentMaps(self, map_type):
    recover_env = False
    if 'DEEPREC_UNIQUE_OP_HASH_MAP' in os.environ:
      recover_env = True
      old_env = os.environ['DEEPREC_UNIQUE_OP_HASH_MAP']

    os.environ['DEEPREC_UNIQUE_OP_HASH_MAP'] = map_type
    self.testInt32()
    self.testInt32OutIdxInt64()

    del os.environ['DEEPREC_UNIQUE_OP_HASH_MAP']
    if recover_env:
      os.environ['DEEPREC_UNIQUE_OP_HASH_MAP'] = old_env

  def testUniqueWithCountsMultiMap(self):
    self.RunUniqueWithCountsWithDifferentMaps('MULTIMAP')

  def testUniqueWithCountsStlMap(self):
    self.RunUniqueWithCountsWithDifferentMaps('STL')

  def testUniqueWithCountsAbslMap(self):
    self.RunUniqueWithCountsWithDifferentMaps('ABSL')

  def testUniqueWithCountsDenseHashMap(self):
    self.RunUniqueWithCountsWithDifferentMaps('GOOGLE')

if __name__ == '__main__':
  test.main()
