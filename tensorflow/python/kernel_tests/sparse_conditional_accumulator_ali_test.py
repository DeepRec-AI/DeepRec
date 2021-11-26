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

import time
import numpy as np
import os

# set environ before tf initializing global varialbes
PreservedKey = 1 << 10;
os.environ["DEEPREC_CONFIG_RAND_64"] = str(PreservedKey)

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


def _indexedslice(ids, dim, size):

  #indices = np.sort(np.random.randint(size, size=ids))
  indices = np.sort(np.random.choice(range(size), ids, replace=False))
  values = np.random.ranf(size=(ids, dim)).astype(np.float32)
  dense_shape = np.array([size, dim])

  return ops.IndexedSlices(indices=indices, values=values, dense_shape=dense_shape)

def _indexedslice_sort(indexedslice):
  indices = indexedslice.indices
  values = indexedslice.values
  dense_shape = indexedslice.dense_shape
  Z = zip(indices, values)
  Z = sorted(Z)
  indices, values = zip(*Z)
  return ops.IndexedSlicesValue(
      indices=indices, values=values, dense_shape=dense_shape)


class IndexedSlicesConditionalAccumulatorTest(test.TestCase):

  def _assertEqual_indexedslices(self, expected_tensor, result):
    self.assertAllEqual(expected_tensor.indices, result.indices)
    self.assertAllClose(expected_tensor.values, result.values)
    if (result.dense_shape is not None and
        expected_tensor.dense_shape is not None):
      self.assertAllEqual(expected_tensor.dense_shape, result.dense_shape)

  def testAccumulatorApplyAndBlockingTake2(self):
    N = 2000
    S = 2000000
    D = 18
    K = 60
    accumulator_types = ('raw', 'multi_map')
    with self.cached_session() as sess:
      inputs = []
      for i in range(K):
        x = _indexedslice(N, D, S)
        #print("input:", x.indices, x.values)
        inputs.append(x)

      results = []
      for accumulator_type in accumulator_types:
        q = data_flow_ops.SparseConditionalAccumulator(
            dtypes_lib.float32, name="Q", shape=(), 
            reduction_type="MEAN",
            accumulator_type=accumulator_type)

        accum_ops = []
        for x in inputs:
          accum_ops.append(q.apply_indexed_slices_grad(x, local_step=i))

        takeg_t = q.take_indexed_slices_grad(K)

        def apply_indexed_slices_grad():
          for accum_op in accum_ops:
            sess.run(accum_op)

        def take_grad():
          results.append(sess.run(takeg_t))

        accum_thread = self.checkedThread(target=apply_indexed_slices_grad)
        takeg_thread = self.checkedThread(target=take_grad)
        accum_thread.start()
        takeg_thread.start()
        accum_thread.join()
        takeg_thread.join()

      for i in range(1, len(accumulator_types)):
        print("check ", accumulator_types[i])
        if accumulator_types[i] == 'multi_map':
          results[i] = _indexedslice_sort(results[i])

        self._assertEqual_indexedslices(results[0], results[i]);

  def testAccumulatorMultiMapWithInvalidIndices(self):
    with self.cached_session() as sess:
      grad = ops.IndexedSlices(indices=[0, PreservedKey], values=[[1.0], [1.0]], dense_shape=[4, 1])
      acc = data_flow_ops.SparseConditionalAccumulator(
          dtypes_lib.float32, name="Q", shape=(),
          reduction_type="MEAN",
          accumulator_type="multi_map")
      apply_op = acc.apply_indexed_slices_grad(grad, local_step=1)
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Input id is preserved key of dense_hash_map, "
          "not supported: " + str(PreservedKey)):
        sess.run(apply_op)


if __name__ == "__main__":
  test.main()
