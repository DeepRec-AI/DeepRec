# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.ops.hash_table import hash_table
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import test

class HashTableEmbeddingTest(test.TestCase):
  def testEmbeddingLookup(self):
    ht = hash_table.DistributedHashTable(
      [4], dtypes.float32,
      partitioner=hash_table.FixedSizeHashTablePartitioner(2),
      initializer=init_ops.ones_initializer(dtypes.float32))
    input_ids = [0, 1, 2, 60000, 60001]
    emb = embedding_ops.embedding_lookup(ht, input_ids, 0)
    with MonitoredTrainingSession('') as sess:
      emb_result = sess.run(emb)
      self.assertAllEqual(emb_result, [[1., 1., 1., 1.]] *5)

  def testEmbeddingLookupSparse(self):
    ht = hash_table.DistributedHashTable(
      [4], dtypes.float32,
      partitioner=hash_table.FixedSizeHashTablePartitioner(2),
      initializer=init_ops.ones_initializer(dtypes.float32))
    input_sparse = sparse_tensor.SparseTensor(
        indices=((0, 0), (0, 1), (1, 0), (3, 0)),
        values=np.array((0, 1, 2, 3)),
        dense_shape=(4, 3))
    emb1 = embedding_ops.embedding_lookup_sparse(ht, input_sparse, None, combiner='mean')
    emb2 = embedding_ops.embedding_lookup_sparse(ht, input_sparse, None, combiner='sum')
    emb3 = embedding_ops.embedding_lookup_sparse(ht, input_sparse, None, combiner='sqrtn')
    emb4 = embedding_ops.embedding_lookup_sparse(ht, input_sparse, None, combiner='tile')
    with MonitoredTrainingSession('') as sess:
      emb_result1 = sess.run(emb1)
      emb_result2 = sess.run(emb2)
      emb_result3 = sess.run(emb3)
      emb_result4 = sess.run(emb4)
      self.assertAllEqual(emb_result1,
                          [[1.,1.,1.,1.],
                           [1.,1.,1.,1.],
                           [0.,0.,0.,0.],
                           [1.,1.,1.,1.]])
      self.assertAllEqual(emb_result2,
                          [[2.,2.,2.,2.],
                           [1.,1.,1.,1.],
                           [0.,0.,0.,0.],
                           [1.,1.,1.,1.]])
      self.assertAllClose(emb_result3,
                          [[1.4142135,1.4142135,1.4142135,1.4142135],
                           [1.,1.,1.,1.],
                           [0.,0.,0.,0.],
                           [1.,1.,1.,1.]])
      self.assertAllEqual(emb_result4,
                          [[1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.],
                           [1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.]])

  def testEmbeddingLookupSparseWithWeight(self):
    ht = hash_table.DistributedHashTable(
      [4], dtypes.float32,
      partitioner=hash_table.FixedSizeHashTablePartitioner(2),
      initializer=init_ops.ones_initializer(dtypes.float32))
    input_sparse = sparse_tensor.SparseTensor(
        indices=((0, 0), (0, 1), (1, 0), (3, 0)),
        values=np.array((0, 1, 2, 3)),
        dense_shape=(4, 3))
    input_weight = sparse_tensor.SparseTensor(
        indices=((0, 0), (0, 1), (1, 0), (3, 0)),
        values=np.array((1., 0.5, 2., 0.)),
        dense_shape=(4, 3))
    emb1 = embedding_ops.safe_embedding_lookup_sparse(ht, input_sparse, input_weight, combiner='mean')
    emb2 = embedding_ops.safe_embedding_lookup_sparse(ht, input_sparse, input_weight, combiner='sum')
    emb3 = embedding_ops.safe_embedding_lookup_sparse(ht, input_sparse, input_weight, combiner='sqrtn')
    emb4 = embedding_ops.safe_embedding_lookup_sparse(ht, input_sparse, input_weight, combiner='tile')
    with MonitoredTrainingSession('') as sess:
      emb_result1 = sess.run(emb1)
      emb_result2 = sess.run(emb2)
      emb_result3 = sess.run(emb3)
      emb_result4 = sess.run(emb4)
      self.assertAllEqual(emb_result1,
                          [[1.,1.,1.,1.],
                           [1.,1.,1.,1.],
                           [0.,0.,0.,0.],
                           [0.,0.,0.,0.]])
      self.assertAllEqual(emb_result2,
                          [[1.5,1.5,1.5,1.5],
                           [2.,2.,2.,2.],
                           [0.,0.,0.,0.],
                           [0.,0.,0.,0.]])
      self.assertAllClose(emb_result3,
                          [[1.341641,1.341641,1.341641,1.341641],
                           [1.,1.,1.,1.],
                           [0.,0.,0.,0.],
                           [0.,0.,0.,0.]])
      self.assertAllEqual(emb_result4,
                          [[1.,1.,1.,1.,0.5,0.5,0.5,0.5,0.,0.,0.,0.],
                           [2.,2.,2.,2.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]])


if __name__ == "__main__":
  test.main()
