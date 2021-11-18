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

import contextlib
import math
import os
import random
import shutil
import tempfile
import time

import numpy as np
import six

from google.protobuf.any_pb2 import Any
from google.protobuf import text_format

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import gen_hash_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.ops import math_ops
from tensorflow.python.training import adagrad
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import saver as saver_module
from tensorflow.python.framework import ops


from tensorflow.python.util import compat
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.hash_table import hash_table
from tensorflow.python.ops.hash_table import hash_filter
from tensorflow.python.ops import embedding_ops
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
from tensorflow.python.ops import gen_hash_ops
from tensorflow.python.training.training_util import get_or_create_global_step
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op

class HashFilterTest(test.TestCase):
  def testGlobalStepFilter(self):
    tensors = []
    for i in range(3):
      offset = i*4
      tensors.append(constant_op.constant([offset, offset+1, offset+2, offset+3], dtype=dtypes.int64))
    dataset = dataset_ops.Dataset.from_tensor_slices(tensors).repeat()
    input_ids = dataset.make_one_shot_iterator().get_next()

    ht = hash_table.DistributedHashTable(
      [4], dtypes.float32,
      partitioner=hash_table.FixedSizeHashTablePartitioner(2),
      initializer=init_ops.ones_initializer(dtypes.float32))

    filter_hook = hash_filter.HashFilterHook(is_chief=True)

    with hash_filter.GlobalStepFilter(1):
      emb = embedding_ops.embedding_lookup(ht, input_ids, 0)

    loss = math_ops.reduce_mean(emb, 0)
    opt = adagrad.AdagradOptimizer(0.1)
    global_step = get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)      
    snapshot = ht.snapshot
    embs = ht.lookup([0,1,2,3,4,5,6,7])
    with MonitoredTrainingSession('', hooks=[filter_hook]) as sess:
      internal_sess = hash_filter._get_internal_sess(sess)
      for i in range(10):
        sess.run(train_op)
      hash_filter.filter_once(sess)
      left_ids = internal_sess.run(snapshot)[0]
      self.assertTrue(np.sort(left_ids).tolist() == [0,1,2,3] or np.sort(left_ids).tolist() == [0,1,2,3,8,9,10,11])
      emb = internal_sess.run(embs)
      self.assertTrue((emb[4:,:] == np.ones((4,4), dtype=np.float32)).all())

  def testL2WeightFilter(self):
    ht = hash_table.DistributedHashTable(
      [4], dtypes.float32,
      partitioner=hash_table.FixedSizeHashTablePartitioner(2),
      initializer=init_ops.ones_initializer(dtypes.float32))

    with hash_filter.L2WeightFilter(1.0):
      emb = embedding_ops.embedding_lookup(ht, [1,2,3,4], 0)

    ids = ht.partitions[0].gen_ids([1,2])
    update_op = gen_hash_ops.tensible_variable_scatter_update(ht.partitions[0].handle, ids, [[0.,0.,0.,0.],[2.,2.,2.,2.]])
    value = ht.lookup([1,2,3,4])
    snapshot = ht.snapshot
    with MonitoredTrainingSession() as sess:
      sess.run(emb)
      sess.run(update_op)
      sess.run(value)
      hash_filter.filter_once(sess)
      ids, keys = sess.run(snapshot)
      self.assertTrue((np.sort(ids).tolist() == [2,3,4]))

if __name__ == "__main__":
  test.main()
