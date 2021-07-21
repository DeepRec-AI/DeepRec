# Copyright (c) 2017, Alibaba Inc.
# All right reserved.
#
# Author: Chen Ding <cnady.dc@alibaba-inc.com>
# Created: 2018/03/26
# Description:
# ==============================================================================

"""Tests for tensorflow.ops.embedding_variable."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.training import ftrl
from tensorflow.python.training import adam_async
from tensorflow.python.training import adagrad
from tensorflow.python.training import adagrad_decay
from tensorflow.python.training import adagrad_decay_v2
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import training_util
from tensorflow.python.ops import variables
from tensorflow.contrib.layers.python.layers import embedding_ops as emb_ops
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader


class EmbeddingVariableTest(test_util.TensorFlowTestCase):

  def testEmbeddingVariableForLookupInt64(self):
    print("testEmbeddingVariableForLookupInt64")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,-7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForExport(self):
    print("testEmbeddingVariableForExport")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(emb)
      print(sess.run([var.export()]))
  
  def testEmbeddingVariableForGetShape(self):
    print("testEmbeddingVariableForGetShape")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    shape = var.total_count()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb]))
      print(sess.run([shape]))

  def testEmbeddingVariableForSaveAndRestore(self):
    print("testEmbeddingVariableForSaveAndRestore")
    checkpoint_directory = self.get_temp_dir()
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3, key_dtype=dtypes.int64,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      save_path = saver.save(sess, os.path.join(checkpoint_directory, "model.ckpt"), global_step=12345)
      print(save_path)
      for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
        print('loading... ', name, shape)

    with self.test_session() as sess:
      saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt-12345"))
      print(sess.run([emb]))

  def testEmbeddingVariableForSparseColumnSharedEmbeddingCol(self):
    columns_list=[]
    columns_list.append(feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.string))
    W = feature_column.shared_embedding_columns(sparse_id_columns=columns_list,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            shared_embedding_name="xxxxx_shared")

    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,0],[2,0],[3,0],[4,0]], values=["aaaa","bbbbb","ccc","4nn","5b"], dense_shape=[5, 5])
    emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=W)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run(init)
      print("init global done")
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb]))

  def testEmbeddingVariableForSparseColumnEmbeddingCol(self):
    columns = feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.int64)
    W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32))

    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=math_ops.cast([1,2,3,4,5], dtypes.int64), dense_shape=[5, 4])

    emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=[W])

    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run(init)
      print("init global done")
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb]))

  def testEmbeddingVariableForShrinkNone(self):
      print("testEmbeddingVariableForShrink")
      var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              steps_to_live=5)
      ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      gs = training_util.get_or_create_global_step()
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        for i in range(10):
          print(sess.run([emb, train_op,loss], feed_dict={'ids:0': 2*i}))

  def testEmbeddingVariableForWeightedSumFromFeatureColumn(self):
    print("testEmbeddingVariableForWeightedSumFromFeatureColumn")
    columns_list=[]
    columns_list.append(feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.string))
    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,0],[2,0],[3,0],[4,0]], values=["aaaa","bbbbb","ccc","4nn","5b"], dense_shape=[5, 5])

    emb, _, _ = feature_column_ops.weighted_sum_from_feature_columns(columns_to_tensors=ids, feature_columns=columns_list, num_outputs=2)

    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run(init)
      print("init global done")
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb]))

  def testEmbeddingVariableForLookupInt64Adagrad(self):
    print("testEmbeddingVariableForLookupInt64Adagrad")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3, key_dtype=dtypes.int64,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    ids = math_ops.cast([0,1,2,5,6,7], dtypes.int64)
    emb = embedding_ops.embedding_lookup(var, ids)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForAdagradDecay(self):
    print("testEmbeddingVariableForAdagradDecay")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForAdagradDecayFilter(self):
    print("testEmbeddingVariableForAdagradDecayFilter")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4), min_freq = 3)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForFtrlFilter(self):
    print("testEmbeddingVariableForFtrlFilter")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4), min_freq = 3)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForAdamAsyncFilter(self):
    print("testEmbeddingVariableForAdamAsynsFilter")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4), min_freq = 3)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adam_async.AdamAsyncOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForGradientDescentFilter(self):
    print("testEmbeddingVariableForGradientDescentFilter")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4), min_freq = 3)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)
  
  def testEmbeddingVariableForAdagradDecayV2Filter(self):
    print("testEmbeddingVariableForAdagradDecayV2Filter")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4), min_freq = 3)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForAdagradDecayV2(self):
    print("testEmbeddingVariableForAdagradDecayV2")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForAdagradDecayStep(self):
    print("testEmbeddingVariableForAdagradDecayStep")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs, accumulator_decay_step=2)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      self.assertEquals(36.0, sess.run([emb, train_op, loss])[2])
      self.assertAlmostEqual(32.444176, sess.run([emb, train_op, loss])[2], delta=1e-05)
      self.assertAlmostEqual(29.847788, sess.run([emb, train_op, loss])[2], delta=1e-05)
      self.assertAlmostEqual(27.74195 , sess.run([emb, train_op, loss])[2], delta=1e-05)
      self.assertAlmostEqual(25.852505, sess.run([emb, train_op, loss])[2], delta=1e-05)

  def testEmbeddingVariableForLookupInt64AdamAsync(self):
    print("testEmbeddingVariableForLookupInt64AdamAsync")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3, key_dtype=dtypes.int64,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    ids = math_ops.cast([0,1,2,5,6,7], dtypes.int64)
    emb = embedding_ops.embedding_lookup(var, ids)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = adam_async.AdamAsyncOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableRestoreSavedModel(self):
    checkpoint_directory = self.get_temp_dir() + "/save_model"
    print("testEmbeddingVariableRestoreSavedModel")
    # build graph
    columns_list=[]
    columns_list.append(feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.string))
    W = feature_column.shared_embedding_columns(sparse_id_columns=columns_list,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            shared_embedding_name="xxxxx_shared")

    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,0],[2,0]], values=["aaaa","bbbbb","ccc"], dense_shape=[3, 5])
    emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=W)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    gs = training_util.get_or_create_global_step()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run(init)
      builder = saved_model_builder.SavedModelBuilder(checkpoint_directory)
      builder.add_meta_graph_and_variables(sess, ['tag_string'])
      builder.save()
    # load savedmodel
    with self.test_session() as sess:
      loader.load(sess, ['tag_string'], checkpoint_directory)

  def testEmbeddingVariableForGeneralConstInitializer(self):
    print("testEmbeddingVariableForGeneralConstInitializer")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,6], dtypes.int64))
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_val = sess.run(emb)
      for i in xrange(2):
        for j in xrange(3):
          self.assertAlmostEqual(1.0, emb_val[i][j], delta=1e-05)

  def testEmbeddingVariableForGeneralRandomInitializer(self):
    print("testEmbeddingVariableForGeneralRandomInitializer")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            #initializer=init_ops.truncated_normal_initializer(dtype=dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,6], dtypes.int64))
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_val = sess.run(emb)
      print(emb_val)
      for i in xrange(3):
        self.assertNotEqual(emb_val[0][i], emb_val[1][i])
        self.assertNotEqual(emb_val[0][i], emb_val[1][i])
        self.assertNotEqual(emb_val[0][i], emb_val[1][i])

  def testEmbeddingVariableForGradientDescent(self):
    print("testEmbeddingVariableForGradientDescent")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForHTPartitionNum(self):
    print("testEmbeddingVariableForHTPartitionNum")
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            ht_partition_num=20)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,-7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

if __name__ == "__main__":
  googletest.main()
