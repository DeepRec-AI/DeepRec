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

# set environ before tf initializing global varialbes
PreservedKey = 1 << 10;
os.environ["DEEPREC_CONFIG_RAND_64"] = str(PreservedKey)

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
from tensorflow.python.ops.hash_table import hash_table
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import adagrad
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import saver as saver_module
from tensorflow.python.util import compat
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops


# @test_util.with_c_api
class HashTableTest(test.TestCase):
  def testGatherAndOptimizer(self):
    with self.test_session(graph=ops_lib.Graph()) as sess:
      ht = hash_table.DistributedHashTable(
        [2,2], dtypes.float32,
        partitioner=hash_table.FixedSizeHashTablePartitioner(5),
        initializer=init_ops.zeros_initializer(dtypes.float32))
      x = variables.Variable(0., dtype=dtypes.float32)
      lookup1 = ht.lookup([0, 1, 2, 60000, 60001])
      lookup2 = ht.lookup([0, 60000, 3])
      lookup3 = ht.lookup([0, 1, 2, 3, 60000, 60001])
      opt = adagrad.AdagradOptimizer(0.1)
      train1 = opt.minimize(lookup1)
      train2 = opt.minimize(lookup2)
      train3 = opt.minimize(x)
      sess.run(variables.global_variables_initializer())
      sess.run(train3)
      adagrad_result1 = sess.run(x)
      sess.run(train3)
      adagrad_result2 = sess.run(x)
      sess.run(train1)
      sess.run(train2)
      rst = sess.run(lookup3)
      print(adagrad_result1, adagrad_result2)
      print(rst)
      self.assertTrue((np.abs(adagrad_result2 - rst[0]) < 1e-5).all())
      self.assertTrue((np.abs(adagrad_result1 - rst[1]) < 1e-5).all())
      self.assertTrue((np.abs(adagrad_result1 - rst[2]) < 1e-5).all())
      self.assertTrue((np.abs(adagrad_result1 - rst[3]) < 1e-5).all())
      self.assertTrue((np.abs(adagrad_result2 - rst[4]) < 1e-5).all())
      self.assertTrue((np.abs(adagrad_result1 - rst[5]) < 1e-5).all())

  def testHashTableTrainable(self):
    ht1 = hash_table.HashTable(
        [2], dtypes.float32, 'table1', init_ops.zeros_initializer(),
        name='table1')
    ht2 = hash_table.HashTable(
        [2], dtypes.float32, 'table2', init_ops.zeros_initializer(),
        trainable=None, name='table2')
    trainables = ops_lib.get_collection(ops_lib.GraphKeys.TRAINABLE_VARIABLES)
    self.assertTrue(ht1 in trainables)
    self.assertTrue(ht2 in trainables)

  def testGetHashTable(self):
    def get_variable(name):
      with variable_scope.variable_scope("get_hashtable_scope",
                                         reuse=variable_scope.AUTO_REUSE) as vs:
        return variable_scope.get_variable(
            name, [2], dtype=dtypes.float32,
            initializer=init_ops.zeros_initializer())

    def get_hashtable(name):
      with variable_scope.variable_scope("get_hashtable_scope",
                                         reuse=variable_scope.AUTO_REUSE) as vs:
        return variable_scope.get_hash_table(
            name, [2], dtype=dtypes.float32,
            initializer=init_ops.zeros_initializer())

    def get_distribute_hashtable(name):
      with variable_scope.variable_scope("get_hashtable_scope",
                                         reuse=variable_scope.AUTO_REUSE) as vs:
        return variable_scope.get_hash_table(
            name, [2,2], dtype=dtypes.float32,
            partitioner=hash_table.FixedSizeHashTablePartitioner(2),
            initializer=init_ops.zeros_initializer(dtypes.float32))

    hashtable_a = get_hashtable("a")
    hashtable_b = get_hashtable("a")
    self.assertTrue(hashtable_a.name == hashtable_b.name)
    self.assertTrue(hashtable_a.name.startswith("get_hashtable_scope"))
    distribute_a = get_distribute_hashtable("b")
    distribute_b = get_distribute_hashtable("b")
    self.assertTrue(distribute_a.name == distribute_b.name)
    self.assertTrue(distribute_a.name.startswith("get_hashtable_scope"))
    self.assertTrue(len(distribute_a.partitions) == len(distribute_b.partitions))
    self.assertTrue(distribute_a.partitions[0].name == distribute_b.partitions[0].name)
    variable_c = get_variable("c")
    with self.assertRaisesRegexp(ValueError, "Trying to reuse variable"):
      failed_var = get_variable("a")
    with self.assertRaisesRegexp(ValueError, "Trying to reuse hashtable"):
      failed_hashtable = get_hashtable("c")

  def testHashTableScatterOp(self):
    with self.test_session(graph=ops_lib.Graph()) as sess:
      ht = hash_table.HashTable(
          [2], dtypes.float32, 'table1', init_ops.zeros_initializer(),
          name='table1'
          )
      update_op = ht.scatter_update([0, 1, 5], [[0.,0.],[1.,1.],[5.,5.]])
      lookup_update = ht.lookup([0, 1, 2, 5])
      add_op = ht.scatter_add([0, 1, 5], [[1.,1.],[-1.,-1.],[5.,-5.]])
      lookup_add = ht.lookup([0, 1, 2, 5])
      sub_op = ht.scatter_sub([0, 1, 2], [[1.,1.],[-1.,-1.],[1.,-1.]])
      lookup_sub = ht.lookup([0, 1, 2, 5])
      mul_op = ht.scatter_mul([0, 1, 5], [[2.,0.5],[-1.,-2.],[0.1,3.333]])
      lookup_mul = ht.lookup([0, 1, 2, 5])
      div_op = ht.scatter_div([0, 1, 5], [[2.,0.5],[-0.1,2.],[10,3.333]])
      lookup_div = ht.lookup([0, 1, 2, 5])

      sess.run(variables.global_variables_initializer())
      sess.run(update_op)
      update_re = sess.run(lookup_update)
      sess.run(add_op)
      add_re = sess.run(lookup_add)
      sess.run(sub_op)
      sub_re = sess.run(lookup_sub)
      sess.run(mul_op)
      mul_re = sess.run(lookup_mul)
      sess.run(div_op)
      div_re = sess.run(lookup_div)
      self.assertAllClose(update_re, [[0.,0.],[1.,1.],[0.,0.],[5.,5.]])
      self.assertAllClose(add_re, [[1.,1.],[0.,0.],[0.,0.],[10.,0.]])
      self.assertAllClose(sub_re, [[0.,0.],[1.,1.],[-1.,1.],[10.,0.]])
      self.assertAllClose(mul_re, [[0.,0.],[-1.,-2.],[-1.,1.],[1.,0.]])
      self.assertAllClose(div_re, [[0.,0.],[10.,-1.],[-1.,1.],[0.1,0.]])

  def testDistributedHashTableScatterOp(self):
    with self.test_session(graph=ops_lib.Graph()) as sess:
      ht = hash_table.DistributedHashTable(
          [2], dtypes.float32,
          partitioner=hash_table.FixedSizeHashTablePartitioner(5),
          initializer=init_ops.zeros_initializer(),
          name='table1'
          )
      update_op = ht.scatter_update([0, 1, 5], [[0.,0.],[1.,1.],[5.,5.]])
      lookup_update = ht.lookup([0, 1, 2, 5])
      add_op = ht.scatter_add([0, 1, 5], [[1.,1.],[-1.,-1.],[5.,-5.]])
      lookup_add = ht.lookup([0, 1, 2, 5])
      sub_op = ht.scatter_sub([0, 1, 2], [[1.,1.],[-1.,-1.],[1.,-1.]])
      lookup_sub = ht.lookup([0, 1, 2, 5])
      mul_op = ht.scatter_mul([0, 1, 5], [[2.,0.5],[-1.,-2.],[0.1,3.333]])
      lookup_mul = ht.lookup([0, 1, 2, 5])
      div_op = ht.scatter_div([0, 1, 5], [[2.,0.5],[-0.1,2.],[10,3.333]])
      lookup_div = ht.lookup([0, 1, 2, 5])

      sess.run(variables.global_variables_initializer())
      sess.run(update_op)
      update_re = sess.run(lookup_update)
      sess.run(add_op)
      add_re = sess.run(lookup_add)
      sess.run(sub_op)
      sub_re = sess.run(lookup_sub)
      sess.run(mul_op)
      mul_re = sess.run(lookup_mul)
      sess.run(div_op)
      div_re = sess.run(lookup_div)
      self.assertAllClose(update_re, [[0.,0.],[1.,1.],[0.,0.],[5.,5.]])
      self.assertAllClose(add_re, [[1.,1.],[0.,0.],[0.,0.],[10.,0.]])
      self.assertAllClose(sub_re, [[0.,0.],[1.,1.],[-1.,1.],[10.,0.]])
      self.assertAllClose(mul_re, [[0.,0.],[-1.,-2.],[-1.,1.],[1.,0.]])
      self.assertAllClose(div_re, [[0.,0.],[10.,-1.],[-1.,1.],[0.1,0.]])

  def testLookupIllegalIds(self):
    with self.test_session(graph=ops_lib.Graph()) as sess:
      ht = hash_table.DistributedHashTable(
        [2,2], dtypes.float32,
        partitioner=hash_table.FixedSizeHashTablePartitioner(5),
        initializer=init_ops.zeros_initializer(dtypes.float32))
      lookup1 = ht.lookup([0, 1, 2, 60000, 60001])
      lookup2 = ht.lookup([PreservedKey])
      lookup3 = ht.lookup([PreservedKey+1])
      sess.run(variables.global_variables_initializer())
      sess.run(lookup1)
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          r"Input key is preserved key of dense_hash_map, "
          "not supported: " + str(PreservedKey)):
        sess.run(lookup2)
      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          r"Input key is preserved key of dense_hash_map, "
          "not supported: " + str(PreservedKey+1)):
        sess.run(lookup3)

if __name__ == "__main__":
  test.main()
