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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.hash_table import hash_table
from tensorflow.python.ops.hash_table import admit_strategy
from tensorflow.python.platform import test

def admit_strategy_factory(hash_table):
  return admit_strategy.BloomFilterAdmitStrategy(
      10, slicer=hash_table.slicer, hash_table=hash_table).handle


class AdmitStrategyTest(test.TestCase):
  def testBloomFilterLookup(self):
    with self.test_session(graph=ops_lib.Graph()) as sess:
      ht = hash_table.DistributedHashTable(
          [2], dtypes.float32,
          partitioner=hash_table.FixedSizeHashTablePartitioner(5),
          initializer=init_ops.ones_initializer(dtypes.float32))
      p_keys = array_ops.placeholder(dtypes.int64, shape=[None], name="keys")
      p_counts = array_ops.placeholder(dtypes.int32, shape=[None], name="counts")
      lookup = ht.lookup(p_keys, admit_strategy_factory, p_counts)
      sess.run(variables.global_variables_initializer())

      keys = np.array([0, 1, 2, -10000, 50000], dtype='int64')
      counts = np.array([6, 10, 2, 2, 3], dtype='int32')
      result = sess.run(lookup, feed_dict={p_keys: keys, p_counts: counts})
      expect_result = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0],
          [0.0, 0.0]], dtype='float32')
      self.assertTrue(np.allclose(result, expect_result))

      keys = np.array([2, 0, -10000], dtype='int64')
      counts = np.array([8, 10000, 7], dtype='int32')
      result = sess.run(lookup, feed_dict={p_keys: keys, p_counts: counts})
      expect_result = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype='float32')
      self.assertTrue(np.allclose(result, expect_result))

if __name__ == '__main__':
  test.main()
