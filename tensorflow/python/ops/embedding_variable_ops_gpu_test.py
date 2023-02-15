# Copyright (c) 2022, NVIDIA.
# All right reserved.
#
# ==============================================================================

"""Tests for tensorflow.ops.embedding_variable GPU version."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import googletest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import sparse_tensor
from tensorflow.core.framework.embedding import config_pb2
from tensorflow.python.training import ftrl
from tensorflow.python.training import adam
from tensorflow.python.training import adam_async
from tensorflow.python.training import adagrad
from tensorflow.python.training import adagrad_decay
from tensorflow.python.training import adagrad_decay_v2
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import training_util
from tensorflow.python.training import weight_decay_optimizers
from tensorflow.python.ops import variables
from tensorflow.contrib.layers.python.layers import embedding_ops as emb_ops
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader


class EmbeddingVariableGpuTest(test_util.TensorFlowTestCase):
  def testDynamicDimensionEmbeddingVariable(self):
    print("testDynamicDimensionEmbeddingVariable")
    with ops.device('/gpu:0'):
      def runTestAdagrad(self, var, g):
        if isinstance(var, kv_variable_ops.EmbeddingVariable):
          emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        else:
          emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64), blocknums=[2,2,2,2,2,2])
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
    with ops.device('/gpu:0'), ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            initializer=init_ops.ones_initializer(dtypes.float32),
            embedding_dim = 8,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      emb1 = runTestAdagrad(self, emb_var, g)
    with ops.device('/gpu:0'), ops.Graph().as_default() as g:
      var =  variable_scope.get_dynamic_dimension_embedding_variable("var_dist",
                                                                    embedding_block_dimension=4,
                                                                    embedding_block_num=2,
                                                                    storage_type=config_pb2.StorageType.HBM,
                                                                    initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdagrad(self, var, g)
    for i in range(0, 6):
      for j in range(0, 8):
        self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testDynamicEmbeddingVariableForInitFromProto(self):
    print("testDynamicEmbeddingVariableForInitFromProto")
    with ops.device('/gpu:0'):
      embedding = variable_scope.get_dynamic_dimension_embedding_variable("var_dist",
                                                                      embedding_block_dimension=4,
                                                                      embedding_block_num=2,
                                                                      storage_type=config_pb2.StorageType.HBM,
                                                                      initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(embedding, math_ops.cast([0,1,2,5,6,7], dtypes.int64), blocknums=[2,2,2,2,2,2])
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    graph = ops.get_default_graph()
    meta_graph_def = saver_module.export_meta_graph()
    ops.reset_default_graph()
    with self.test_session() as sess:
      res = saver_module.import_meta_graph(meta_graph_def)

  def testEmbeddingVariableForInitFromProto(self):
    print("testEmbeddingVariableForInitFromProto")
    with ops.device('/gpu:0'):
      embedding = variable_scope.get_embedding_variable("var_dist",
          embedding_dim=6,
          initializer=init_ops.ones_initializer,
          steps_to_live = 4,
          ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)),
          partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(embedding, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    graph = ops.get_default_graph()
    meta_graph_def = saver_module.export_meta_graph()
    ops.reset_default_graph()
    with self.test_session() as sess:
      res = saver_module.import_meta_graph(meta_graph_def)

  def testEmbeddingVariableForLookupInt64(self):
    print("testEmbeddingVariableForLookupInt64")
    with ops.device("/gpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,-7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session(force_gpu=True) as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  '''
  def testEmbeddingVariableForExport(self):
    print("testEmbeddingVariableForExport")
    ev_config = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=1),
                                                  storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM))
    with ops.device("/gpu:0"):
      var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
              initializer=init_ops.ones_initializer(dtypes.float32), steps_to_live=10000, ev_option=ev_config)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    init = variables.global_variables_initializer()
    keys, values, versions, freqs = var.export()
    with self.test_session(force_gpu=True) as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(emb)
      sess.run(emb)
      sess.run(emb)
      fetches = sess.run([keys, values, versions, freqs])
      fetches[0].sort()
      print(fetches)
      self.assertAllEqual([0, 1, 2, 5, 6, 7], fetches[0])
      self.assertAllEqual([[1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.],
                           [1., 1., 1.]], fetches[1])
      # self.assertAllEqual([0, 0, 0, 0, 0, 0], fetches[2])
      # self.assertAllEqual([1, 1, 1, 1, 1, 1], fetches[3])
  '''

  def testEmbeddingVariableForGetShape(self):
    print("testEmbeddingVariableForGetShape")
    with ops.device("/gpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)),
              initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    shape = var.total_count()
    init = variables.global_variables_initializer()
    with self.test_session(force_gpu=True) as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([emb])
      # Unimplement GPUHashMapKV::Size() {return 0;}
      self.assertAllEqual([0, 3], sess.run(shape))

  def testEmbeddingVariableForSparseColumnSharedEmbeddingCol(self):
    columns_list=[]
    columns_list.append(feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.string,
        ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM))))
    with ops.device("/gpu:0"):
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

    with self.test_session(force_gpu=True) as sess:
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
    with ops.device("/gpu:0"):
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

    with self.test_session(force_gpu=True) as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb1, top, l = sess.run([emb, train_op, loss])
      for val1 in emb1.tolist():
        for val in val1:
          self.assertEqual(val, .0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for index, val1 in enumerate(emb1.tolist()):
        if index < 7:
          for val in val1:
            self.assertNotEqual(val, 1.0)
        else:
          for val in val1:
            self.assertEqual(val, .0)

  def testEmbeddingVariableForSparseColumnEmbeddingCol(self):
    columns = feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.int64,
        ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)))
    with ops.device("/gpu:0"):
      W = feature_column.embedding_column(sparse_id_column=columns,
              dimension=3,
              initializer=init_ops.ones_initializer(dtypes.float32))

    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,3]], values=math_ops.cast([1,2,3,4,5], dtypes.int64), dense_shape=[5, 4])

    emb = feature_column_ops.input_from_feature_columns(columns_to_tensors=ids, feature_columns=[W])

    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    with self.test_session(force_gpu=True) as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run(init)
      print("init global done")
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))
      print(sess.run([emb, train_op,loss]))

  def testEmbeddingVariableForAdagrad(self):
    print("testEmbeddingVariableForAdagrad")
    def runTestAdagrad(self, var):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(force_gpu=True) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    with ops.device("/gpu:0"):
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
    with ops.device("/cpu:0"):
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
    emb1 = runTestAdagrad(self, emb_var)
    emb2 = runTestAdagrad(self, var)
    for i in range(0, 6):
      for j in range(0, 3):
        self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForFtrl(self):
    print("testEmbeddingVariableForFtrl")
    def runTestFtrl(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g, force_gpu=True) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r
    with ops.Graph().as_default() as g:
      with ops.device('/gpu:0'):
        emb_var = variable_scope.get_embedding_variable("var_1", embedding_dim=3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      with ops.device('/cpu:0'):
        emb_var2 = variable_scope.get_embedding_variable("var_2", embedding_dim=3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM)),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      emb1 = runTestFtrl(self, emb_var, g)
      emb2 = runTestFtrl(self, emb_var2, g)
      for i in range(0, 6):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForGeneralConstInitializer(self):
    print("testEmbeddingVariableForGeneralConstInitializer")
    with ops.device('/gpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,6], dtypes.int64))
    init = variables.global_variables_initializer()
    with self.test_session(force_gpu=True) as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_val = sess.run(emb)
      for i in range(2):
        for j in range(3):
          self.assertAlmostEqual(1.0, emb_val[i][j], delta=1e-05)

  def testEmbeddingVariableForGeneralRandomInitializer(self):
    print("testEmbeddingVariableForGeneralRandomInitializer")
    with ops.device('/gpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              #initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,6], dtypes.int64))
    init = variables.global_variables_initializer()
    with self.test_session(force_gpu=True) as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_val = sess.run(emb)
      print(emb_val)
      for i in range(3):
        self.assertNotEqual(emb_val[0][i], emb_val[1][i])
        self.assertNotEqual(emb_val[0][i], emb_val[1][i])
        self.assertNotEqual(emb_val[0][i], emb_val[1][i])

  def testEVInitializerWithKeyFetch(self):
    print("testEVInitializerWithKeyFetch")
    with ops.Graph().as_default() as g:
      with ops.device('/gpu:0'):
        var = variable_scope.get_variable("var", shape=[8,3],
                                          initializer=init_ops.glorot_uniform_initializer(seed = 1))
        init_opt = variables.InitializerOption(initializer=init_ops.glorot_uniform_initializer(seed = 1),
                                          default_value_dim=8)
        ev_option = variables.EmbeddingVariableOption(init_option=init_opt,
             storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM))
        emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim=3,
                                                        ev_option=ev_option)
        var_emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,3,4,5,6,7], dtypes.int64))
        emb_emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([0,1,2,5,6,7,8,9,10], dtypes.int64))
        init = variables.global_variables_initializer()
        with self.test_session(graph=g, force_gpu=True) as sess:
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
          sess.run([init])
          emb1 = sess.run(var_emb)
          emb2 = sess.run(emb_emb)
          self.assertListEqual(emb1.tolist()[0], emb2.tolist()[0])
          self.assertListEqual(emb1.tolist()[1], emb2.tolist()[1])
          self.assertListEqual(emb1.tolist()[2], emb2.tolist()[2])
          self.assertListEqual(emb1.tolist()[5], emb2.tolist()[3])
          self.assertListEqual(emb1.tolist()[6], emb2.tolist()[4])
          self.assertListEqual(emb1.tolist()[7], emb2.tolist()[5])
          self.assertListEqual(emb1.tolist()[0], emb2.tolist()[6])
          self.assertListEqual(emb1.tolist()[1], emb2.tolist()[7])
          self.assertListEqual(emb1.tolist()[2], emb2.tolist()[8])

  def testEVInitializer(self):
    def runTest(self, var, g):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g, force_gpu=True) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        return r
    print("testEVInitializer")
    with ops.Graph().as_default() as g:
      with ops.device('/gpu:0'):
        init = variables.InitializerOption(default_value_dim=8192)
        ev_option = variables.EmbeddingVariableOption(init_option = init,
            storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM))
        emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim = 6,
                                          initializer=init_ops.glorot_uniform_initializer(seed = 3),
                                          ev_option = ev_option)
      with ops.device('/cpu:0'):
        var = variable_scope.get_variable("var", shape=[8192, 6],
                                    initializer=init_ops.glorot_uniform_initializer(seed = 3))
      emb1 = runTest(self, emb_var, g)
      emb2 = runTest(self, var, g)

      for i in range(0, 6):
        for j in range(0, 6):
          self.assertAllClose(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testGPUFbjOpt(self):
    print("testGPUFbjOpt")
    os.environ["TF_EMBEDDING_FBJ_OPT"] = "True"
    def runTestAdagrad(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      if isinstance(var, kv_variable_ops.EmbeddingVariable):
        tires = kv_variable_ops.lookup_tier(emb_var,
                    math_ops.cast([1,2,3,4,5,6], dtypes.int64))
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[1,2,3]})
        sess.run([train_op], {ids:[1,2,4]})
        sess.run([train_op], {ids:[1,2,2]})
        sess.run([train_op], {ids:[1,2,5]})

        if isinstance(var, kv_variable_ops.EmbeddingVariable):
          result = sess.run(tires)
          for i in range(0, 6):
            if i == 2:
              self.assertEqual(result[i], 1)
            elif i == 5:
              self.assertEqual(result[i], -1)
            else:
              self.assertEqual(result[i], 0)

        sess.run([train_op], {ids:[3, 5]})
        sess.run([train_op], {ids:[4]})
        r1 = sess.run(emb, {ids:[1,2,4,5]})
        r2 = sess.run(emb, {ids:[3]})
        r = r1.tolist() + r2.tolist()
        return r

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.HBM_DRAM,
                        storage_size=[1024])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = ev_option)
      emb1 = runTestAdagrad(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2",
                shape=[100, 30],
                initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdagrad(self, var, g)

    for i in range(0, 5):
      for j in range(0, 30):
        self.assertAllCloseAccordingToType(emb1[i][j], emb2[i][j])

    del os.environ["TF_EMBEDDING_FBJ_OPT"]

  def testEmbeddingVariableForHBMandDRAMAdamWithFbjOpt(self):
    print("testEmbeddingVariableForHBMandDRAMAdamWithFbjOpt")
    os.environ["TF_EMBEDDING_FBJ_OPT"] = "True"
    def runTestAdam(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adam.AdamOptimizer(0.01)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[1,2,3]})
        sess.run([train_op], {ids:[1,2,4]})
        sess.run([train_op], {ids:[1,2,2]})
        sess.run([train_op], {ids:[1,2,5]})
        r = sess.run(emb, {ids:[1,2,3,4,5]})
        return r

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.HBM_DRAM,
                        storage_size=[1024 * 1024])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = ev_option)
      emb1 = runTestAdam(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2",
                shape=[100, 30],
                initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdam(self, var, g)

    for i in range(0, 5):
      for j in range(0, 30):
        self.assertAllCloseAccordingToType(emb1[i][j], emb2[i][j])
    del os.environ["TF_EMBEDDING_FBJ_OPT"]
  
  def testEmbeddingVariableForHBMandDRAMAdam(self):
    print("testEmbeddingVariableForHBMandDRAMAdam")
    def runTestAdam(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adam.AdamOptimizer(0.01)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[1,2,3]})
        sess.run([train_op], {ids:[1,2,4]})
        sess.run([train_op], {ids:[1,2,2]})
        sess.run([train_op], {ids:[1,2,5]})
        r = sess.run(emb, {ids:[1,2,3,4,5]})
        return r

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.HBM_DRAM,
                        storage_size=[1024 * 1024])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = ev_option)
      emb1 = runTestAdam(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2",
                shape=[100, 30],
                initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdam(self, var, g)

    for i in range(0, 5):
      for j in range(0, 30):
        self.assertAllCloseAccordingToType(emb1[i][j], emb2[i][j])

  def testEmbeddingVariableForHBMandDRAMAdamAsyncWithFbjOpt(self):
    print("testEmbeddingVariableForHBMandDRAMAdamAsyncWithFbjOpt")
    os.environ["TF_EMBEDDING_FBJ_OPT"] = "True"
    def runTestAdam(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adam_async.AdamAsyncOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[1,2,3]})
        sess.run([train_op], {ids:[1,2,4]})
        sess.run([train_op], {ids:[1,2,2]})
        sess.run([train_op], {ids:[1,2,5]})
        r = sess.run(emb, {ids:[1,2,3,4,5]})
        return r

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.HBM_DRAM,
                        storage_size=[1024 * 1024])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = ev_option)
      emb1 = runTestAdam(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2",
                shape=[100, 30],
                initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdam(self, var, g)

    for i in range(0, 5):
      for j in range(0, 30):
        self.assertAllCloseAccordingToType(emb1[i][j], emb2[i][j])
    del os.environ["TF_EMBEDDING_FBJ_OPT"]

  def testEmbeddingVariableForHBMandDRAMAdamAsync(self):
    print("testEmbeddingVariableForHBMandDRAMAdamAsync")
    def runTestAdamAsync(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adam_async.AdamAsyncOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[1,2,3]})
        sess.run([train_op], {ids:[1,2,4]})
        sess.run([train_op], {ids:[1,2,2]})
        sess.run([train_op], {ids:[1,2,5]})
        r = sess.run(emb, {ids:[1,2,3,4,5]})
        return r

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.HBM_DRAM,
                        storage_size=[1024 * 1024])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = ev_option)
      emb1 = runTestAdamAsync(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2",
                shape=[100, 30],
                initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdamAsync(self, var, g)

    for i in range(0, 5):
      for j in range(0, 30):
        self.assertAllCloseAccordingToType(emb1[i][j], emb2[i][j])

  def testEmbeddingVariableForHBMandDRAMAdamWWithFbjOpt(self):
    print("testEmbeddingVariableForHBMandDRAMAdamWWithFbjOpt")
    os.environ["TF_EMBEDDING_FBJ_OPT"] = "True"
    def runTestAdam(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = weight_decay_optimizers.AdamWOptimizer(0.01)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[1,2,3]})
        sess.run([train_op], {ids:[1,2,4]})
        sess.run([train_op], {ids:[1,2,2]})
        sess.run([train_op], {ids:[1,2,5]})
        r = sess.run(emb, {ids:[1,2,3,4,5]})
        return r

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.HBM_DRAM,
                        storage_size=[1024 * 1024])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = ev_option)
      emb1 = runTestAdam(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2",
                shape=[100, 30],
                initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdam(self, var, g)

    for i in range(0, 5):
      for j in range(0, 30):
        self.assertAllCloseAccordingToType(emb1[i][j], emb2[i][j])
    del os.environ["TF_EMBEDDING_FBJ_OPT"]


  def testEmbeddingVariableForHBMandDRAMAdamW(self):
    print("testEmbeddingVariableForHBMandDRAMAdamW")
    def runTestAdamW(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = weight_decay_optimizers.AdamWOptimizer(0.01)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[1,2,3]})
        sess.run([train_op], {ids:[1,2,4]})
        sess.run([train_op], {ids:[1,2,2]})
        sess.run([train_op], {ids:[1,2,5]})
        r = sess.run(emb, {ids:[1,2,3,4,5]})
        return r

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.HBM_DRAM,
                        storage_size=[1024 * 1024])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = ev_option)
      emb1 = runTestAdamW(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2",
                shape=[100, 30],
                initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdamW(self, var, g)

    for i in range(0, 5):
      for j in range(0, 30):
        self.assertAllCloseAccordingToType(emb1[i][j], emb2[i][j])

if __name__ == "__main__":
  googletest.main()
