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
from tensorflow.python.training import adam
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
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
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

  def testEmbeddingVariableForLookupInt32(self):
    print("testEmbeddingVariableForLookupInt32")
    checkpoint_directory = self.get_temp_dir()
    var = variable_scope.get_embedding_variable("var_1",
                                                embedding_dim = 3,
                                                key_dtype=dtypes.int32,
                                                initializer=init_ops.ones_initializer(dtypes.float32),
                                                partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,-7], dtypes.int32))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([train_op])
      model_path = os.path.join(checkpoint_directory, "model.ckpt")
      save_path = saver.save(sess, model_path, global_step=12345)
      saver.restore(sess, save_path)

  def testEmbeddingVariableForExport(self):
    print("testEmbeddingVariableForExport")
    ev_config = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=1))
    var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32), steps_to_live=10000, ev_option=ev_config)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    init = variables.global_variables_initializer()
    keys, values, versions, freqs = var.export()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(emb)
      sess.run(emb)
      sess.run(emb)
      fetches = sess.run([keys, values, versions, freqs])
      print(fetches)
      self.assertAllEqual([0, 1, 2, 5, 6, 7], fetches[0])
      self.assertAllEqual([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]], fetches[1])
      self.assertAllEqual([0, 0, 0, 0, 0, 0], fetches[2])
      self.assertAllEqual([1, 1, 1, 1, 1, 1], fetches[3])

  def testEmbeddingVariableForGetShape(self):
    print("testEmbeddingVariableForGetShape")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    shape = var.total_count()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([emb])
      self.assertAllEqual([6, 3], sess.run(shape))

  def testEmbeddingVariableForMultiHashAdd(self):
    print("testEmbeddingVariableForMultiHashAdd")
    with ops.device('/cpu:0'):
      var1 = variable_scope.get_variable("var_1", shape=[5,6],
                                      initializer=init_ops.ones_initializer(dtypes.float32),
                                      partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      var2 = variable_scope.get_variable("var_2", shape=[3,6],
                                      initializer=init_ops.ones_initializer(dtypes.float32),
                                      partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      ids_Q = math_ops.cast([0//5, 1//5, 2//5 , 4//5, 6//5, 7//5],dtypes.int64)
      ids_R = math_ops.cast([0%3, 1%3, 2%3 , 4%3, 6%3, 7%3],dtypes.int64)
      emb1 =  embedding_ops.embedding_lookup(var1, ids_Q)
      emb2 =  embedding_ops.embedding_lookup(var2, ids_R)
      emb = math_ops.add(emb1, emb2)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)

      ids = math_ops.cast([0, 1, 2, 4, 6, 7], dtypes.int64)
      var_multi = variable_scope.get_multihash_variable("var_multi",
                                         [[5,6],[3,6]],
                                         complementary_strategy="Q-R",
                                         initializer=init_ops.ones_initializer,
                                         partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2)
                                        )
      emb_multi =  embedding_ops.embedding_lookup(var_multi, ids)
      fun_m = math_ops.multiply(emb_multi, 2.0, name='multiply')
      loss_m = math_ops.reduce_sum(fun_m, name='reduce_sum')
      gs_m = training_util.get_or_create_global_step()
      opt_m = adagrad_decay.AdagradDecayOptimizer(0.1, gs_m)
      g_v_m = opt_m.compute_gradients(loss_m)
      train_op_m = opt_m.apply_gradients(g_v_m)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op, train_op_m])
        val_list = sess.run([emb, emb_multi])
        for i in range(ids.shape.as_list()[0]):
          self.assertAllEqual(val_list[0][i], val_list[1][i])

  def testEmbeddingVariableForMultiHashMul(self):
    print("testEmbeddingVariableForMultiHashMul")
    with ops.device('/cpu:0'):
      var1 = variable_scope.get_variable("var_1", shape=[5,6],
                                      initializer=init_ops.ones_initializer(dtypes.float32),
                                      partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      var2 = variable_scope.get_variable("var_2", shape=[3,6],
                                      initializer=init_ops.ones_initializer(dtypes.float32),
                                      partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      ids_Q = math_ops.cast([0//5, 1//5, 2//5 , 4//5, 6//5, 7//5],dtypes.int64)
      ids_R = math_ops.cast([0%3, 1%3, 2%3 , 4%3, 6%3, 7%3],dtypes.int64)
      emb1 =  embedding_ops.embedding_lookup(var1, ids_Q)
      emb2 =  embedding_ops.embedding_lookup(var2, ids_R)
      emb = math_ops.multiply(emb1, emb2)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)

      ids = math_ops.cast([0, 1, 2, 4, 6, 7], dtypes.int64)
      var_multi = variable_scope.get_multihash_variable("var_multi",
                                         [[5,6],[3,6]],
                                         complementary_strategy="Q-R",
                                         operation="mul",
                                         initializer=init_ops.ones_initializer,
                                         partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2)
                                        )
      emb_multi =  embedding_ops.embedding_lookup(var_multi, ids)
      fun_m = math_ops.multiply(emb_multi, 2.0, name='multiply')
      loss_m = math_ops.reduce_sum(fun_m, name='reduce_sum')
      gs_m = training_util.get_or_create_global_step()
      opt_m = adagrad_decay.AdagradDecayOptimizer(0.1, gs_m)
      g_v_m = opt_m.compute_gradients(loss_m)
      train_op_m = opt_m.apply_gradients(g_v_m)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op, train_op_m])
        val_list = sess.run([emb, emb_multi])
        for i in range(ids.shape.as_list()[0]):
          self.assertAllEqual(val_list[0][i], val_list[1][i])

  def testEmbeddingVariableForMultiHashConcat(self):
    print("testEmbeddingVariableForMultiHashConcat")
    with ops.device('/cpu:0'):
      var1 = variable_scope.get_variable("var_1", shape=[5,6],
                                      initializer=init_ops.ones_initializer(dtypes.float32),
                                      partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      var2 = variable_scope.get_variable("var_2", shape=[3,6],
                                      initializer=init_ops.ones_initializer(dtypes.float32),
                                      partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      ids_Q = math_ops.cast([0//5, 1//5, 2//5 , 4//5, 6//5, 7//5],dtypes.int64)
      ids_R = math_ops.cast([0%3, 1%3, 2%3 , 4%3, 6%3, 7%3],dtypes.int64)
      emb1 =  embedding_ops.embedding_lookup(var1, ids_Q)
      emb2 =  embedding_ops.embedding_lookup(var2, ids_R)
      emb = array_ops.concat([emb1, emb2], 1)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)

      ids = math_ops.cast([0, 1, 2, 4, 6, 7], dtypes.int64)
      var_multi = variable_scope.get_multihash_variable("var_multi",
                                         [[5,6],[3,6]],
                                         complementary_strategy="Q-R",
                                         operation="concat",
                                         initializer=init_ops.ones_initializer,
                                         partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2)
                                        )
      emb_multi =  embedding_ops.embedding_lookup(var_multi, ids)
      fun_m = math_ops.multiply(emb_multi, 2.0, name='multiply')
      loss_m = math_ops.reduce_sum(fun_m, name='reduce_sum')
      gs_m = training_util.get_or_create_global_step()
      opt_m = adagrad_decay.AdagradDecayOptimizer(0.1, gs_m)
      g_v_m = opt_m.compute_gradients(loss_m)
      train_op_m = opt_m.apply_gradients(g_v_m)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op, train_op_m])
        val_list = sess.run([emb, emb_multi])
        for i in range(ids.shape.as_list()[0]):
          self.assertAllEqual(val_list[0][i], val_list[1][i])

  def testEmbeddingVariableForSaveAndRestore(self):
    print("testEmbeddingVariableForSaveAndRestore")
    checkpoint_directory = self.get_temp_dir()
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_ori = sess.run(emb)
      emb_ori = sess.run(emb)
      emb_ori = sess.run(emb)
      emb_ori = sess.run(emb)
      save_path = saver.save(sess, os.path.join(checkpoint_directory, "model.ckpt"), global_step=12345)
      print(save_path)
      for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
        print('loading... ', name, shape)

    with self.test_session() as sess:
      saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt-12345"))
      self.assertAllEqual(emb_ori, sess.run(emb))

  def testEmbeddingVariableForL2FeatureEvictionFromContribFeatureColumn(self):
    print("testEmbeddingVariableForL2FeatureEvictionFromContribFeatureColumn")
    checkpoint_directory = self.get_temp_dir()
    evict = variables.L2WeightEvict(l2_weight_threshold=0.9)
    columns = feature_column.sparse_column_with_embedding(
                                        column_name="col_emb",
                                        dtype=dtypes.int64,
                                        ev_option = variables.EmbeddingVariableOption(evict_option=evict))
    W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            combiner="mean")
    ids = {}
    ids["col_emb"] = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([0,0,0,1,1,2], dtypes.int64),
                      dense_shape=[6, 1])
    emb= feature_column_ops.input_from_feature_columns(
           columns_to_tensors=ids, feature_columns=[W])
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_ori = sess.run([emb, train_op])
      save_path = saver.save(sess, os.path.join(checkpoint_directory, "model1.ckpt"), global_step=12345)
    with self.test_session() as sess:
      saver.restore(sess, os.path.join(checkpoint_directory, "model1.ckpt-12345"))
      emb_right = [[0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.7927219, 0.7927219, 0.7927219],
                   [0.7927219, 0.7927219, 0.7927219],
                   [1.0, 1.0, 1.0]]
      emb_ori = sess.run(emb)
      for i in range(6):
        for j in range(3):
          self.assertAlmostEqual(emb_ori[i][j], emb_right[i][j])

  def testEmbeddingVariableForL2FeatureEviction(self):
    print("testEmbeddingVariableForL2FeatureEviction")
    checkpoint_directory = self.get_temp_dir()
    evict = variables.L2WeightEvict(l2_weight_threshold=0.9)
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(evict_option=evict))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,0,0,1,1,2], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_ori = sess.run([emb, train_op])
      save_path = saver.save(sess, os.path.join(checkpoint_directory, "model1.ckpt"), global_step=12345)
      #for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
      #  print('loading... ', name, shape)
    with self.test_session() as sess:
      saver.restore(sess, os.path.join(checkpoint_directory, "model1.ckpt-12345"))
      emb_right = [[0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.8282884, 0.8282884, 0.8282884],
                   [0.7927219, 0.7927219, 0.7927219],
                   [0.7927219, 0.7927219, 0.7927219],
                   [1.0, 1.0, 1.0]]
      emb_ori = sess.run(emb)
      for i in range(6):
        for j in range(3):
          self.assertAlmostEqual(emb_ori[i][j], emb_right[i][j])

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

  def testEmbeddingVariableForFeatureFilterFromContribFeatureColumn(self):
    print("testEmbeddingVariableForFeatureFilterFromContribFeatureColumn")
    columns = feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.int64,
                                                          ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)))
    W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32))
    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0]], values=math_ops.cast([1,1,1,1,2,2,2,3,3,4], dtypes.int64), dense_shape=[10, 1])
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
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val1 in emb1.tolist():
        for val in val1:
          self.assertEqual(val, 1.0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for index, val1 in enumerate(emb1.tolist()):
        if index < 7:
          for val in val1:
            self.assertNotEqual(val, 1.0)
        else:
          for val in val1:
            self.assertEqual(val, 1.0)

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

  def testEmbeddingVariableForShrinkNone(self):
      print("testEmbeddingVariableForShrink")
      var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            steps_to_live = 5,
            initializer=init_ops.ones_initializer(dtypes.float32))
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

  def testEmbeddingVariableForBloomFilterInt64(self):
    print("testEmbeddingVariableForBloomFilterInt64")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 5,
                                      false_positive_probability = 0.01)),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
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

  def testEmbeddingVariableForBloomFilterInt32(self):
    print("testEmbeddingVariableForBloomFilterInt32")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 5,
                                      false_positive_probability = 0.01,
                                      counter_type = dtypes.uint32
                                    )),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
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

  def testEmbeddingVariableForBloomFilterInt8(self):
    print("testEmbeddingVariableForBloomFilterInt8")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 5,
                                      false_positive_probability = 0.01,
                                      counter_type = dtypes.uint8
                                    )),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
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

  def testEmbeddingVariableForBloomFilterInt16(self):
    print("testEmbeddingVariableForBloomFilterInt16")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 5,
                                      false_positive_probability = 0.01,
                                      counter_type = dtypes.uint16
                                    )),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
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

  def testEmbeddingVariableForAdagradDecayFilter(self):
    print("testEmbeddingVariableForAdagradDecayFilter")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
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
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    #var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
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
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
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
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
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
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
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

  def testEmbeddingVariableForAdamFilter(self):
    print("testEmbeddingVariableForAdamFilter")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adam.AdamOptimizer(0.1, gs)
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

  def testEmbeddingVariableForGradientDescent(self):
    print("testEmbeddingVariableForGradientDescent")
    with ops.device('/cpu:0'):
      def runTestGradientDescent(self, var):
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
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestGradientDescent(self, emb_var)
      emb2 = runTestGradientDescent(self, var)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForAdagrad(self):
    print("testEmbeddingVariableForAdagrad")
    with ops.device('/cpu:0'):
      def runTestAdagrad(self, var):
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v)
        init = variables.global_variables_initializer()
        with self.test_session() as sess:
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
          sess.run([init])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var)
      emb2 = runTestAdagrad(self, var)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForAdagradDecay(self):
    print("testEmbeddingVariableForAdagradDecay")
    with ops.device('/cpu:0'):
      def runTestAdagradDecay(self, var):
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
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r
      emb_var = variable_scope.get_embedding_variable("var_1",
            initializer=init_ops.ones_initializer(dtypes.float32),
            embedding_dim = 3,
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagradDecay(self, emb_var)
      emb2 = runTestAdagradDecay(self, var)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForAdagradDecayV2(self):
    print("testEmbeddingVariableForAdagradDecayV2")
    with ops.device('/cpu:0'):
      def runTestAdagradDecayV2(self, var):
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
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagradDecayV2(self, emb_var)
      emb2 = runTestAdagradDecayV2(self, var)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForAdam(self):
    print("testEmbeddingVariableForAdam")
    with ops.device('/cpu:0'):
      def runTestAdam(self, var):
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        fun = math_ops.multiply(emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adam.AdamOptimizer(0.1, gs)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v)
        init = variables.global_variables_initializer()
        with self.test_session() as sess:
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
          sess.run([init])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r = sess.run(emb)
          return r
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdam(self, emb_var)
      emb2 = runTestAdam(self, var)

      print(emb1.tolist())
      print(emb2.tolist())
      for i in range(0, 6):
        for j in range(0, 3):
          self.assertAlmostEqual(emb1.tolist()[i][j], emb2.tolist()[i][j], delta=1e-05)

  def testEmbeddingVariableForAdamAsync(self):
    print("testEmbeddingVariableForAdamAsync")
    with ops.device('/cpu:0'):
      def runTestAdamAsync(self, var):
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
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
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      var = variable_scope.get_variable("var_2", shape=[100, 3],
            initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdamAsync(self, emb_var)
      emb2 = runTestAdamAsync(self, var)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  '''
  def testEmbeddingVariableForFtrl(self):
    print("testEmbeddingVariableForFtrl")
    with ops.device('/cpu:0'):
      def runTestAdam(self, var):
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
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
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r
      emb_var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdam(self, emb_var)
      emb2 = runTestAdam(self, var)

      #for i in range(0, 6):
      #  for j in range(0, 3):
      #    self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])
  '''
  def testEmbeddingVariableForAdagradDecayStep(self):
    print("testEmbeddingVariableForAdagradDecayStep")
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
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
      self.assertEqual(36.0, sess.run([emb, train_op, loss])[2])
      self.assertAlmostEqual(32.444176, sess.run([emb, train_op, loss])[2], delta=1e-05)
      self.assertAlmostEqual(29.847788, sess.run([emb, train_op, loss])[2], delta=1e-05)
      self.assertAlmostEqual(27.74195 , sess.run([emb, train_op, loss])[2], delta=1e-05)
      self.assertAlmostEqual(25.852505, sess.run([emb, train_op, loss])[2], delta=1e-05)

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
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
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
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            #initializer=init_ops.ones_initializer(dtypes.float32),
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

  def testEmbeddingVariableForHTPartitionNum(self):
    print("testEmbeddingVariableForHTPartitionNum")
    ev_option = variables.EmbeddingVariableOption(ht_partition_num=20)
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            ev_option=ev_option)
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

  def testEmbeddingVariableForLayout(self):
    print("testEmbeddingVariableForLayout")
    def runTestAdagrad(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    with ops.device('/cpu:0'), ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

    with ops.device('/cpu:0'), ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5)
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

    with ops.device('/cpu:0'), ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=5)))
      emb1 = runTestAdagrad(self, emb_var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], 1.0)

  def testEmbeddingVariableForDRAM(self):
    print("testEmbeddingVariableForDRAM")
    def runTestAdagrad(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r

    from tensorflow.core.framework.embedding import config_pb2
    with ops.device('/cpu:0'), ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM)))
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTestAdagrad(self, emb_var, g)
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])


if __name__ == "__main__":
  googletest.main()
