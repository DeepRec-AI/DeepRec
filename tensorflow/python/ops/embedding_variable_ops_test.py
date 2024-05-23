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

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.check_ops import assert_equal
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
from tensorflow.python.training import weight_decay_optimizers
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import training_util
from tensorflow.python.ops import variables
from tensorflow.contrib.layers.python.layers import embedding_ops as emb_ops
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.python.feature_column import feature_column as feature_column_v1
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import loader
from tensorflow.core.protobuf import config_pb2 as config_pb3
from tensorflow.python.platform import tf_logging as logging
import time
import random

class EmbeddingVariableTest(test_util.TensorFlowTestCase):
  def _CreateOptimizer(self, optimizer):
    if optimizer == "Adagrad":
      return adagrad.AdagradOptimizer(0.1)
    elif optimizer == "AdagradDecay":
      gs = training_util.get_or_create_global_step()
      return adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    elif optimizer == "AdagradDecayV2":
      gs = training_util.get_or_create_global_step()
      return adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs)
    elif optimizer == "Adam":
      return adam.AdamOptimizer(0.1)
    elif optimizer == "AdamAsync":
      return adam_async.AdamAsyncOptimizer(0.1)
    elif optimizer == "FTRL":
      return ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    elif optimizer == "GradientDescent":
      return gradient_descent.GradientDescentOptimizer(0.1)
    elif optimizer == "AdamW":
      return weight_decay_optimizers.AdamWOptimizer(0.01)
    else:
      logging.fatal("Optimizer name is invalid")

  def _OpitmizerTestTemplate(self, optimizer):
    def runTest(self, var):
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      opt = self._CreateOptimizer(optimizer)
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
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_variable("var_2", shape=[100, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      emb1 = runTest(self, var)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
      emb2 = runTest(self, emb_var)

    for i in range(0, 6):
      for j in range(0, 3):
        self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def _CounterFilterTestTemplate(self, optimizer):
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,1,1,1,2,2,2,3,3,4], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = self._CreateOptimizer(optimizer)
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
            self.assertEqual(val, .0)
        emb1, top, l = sess.run([emb, train_op, loss])
        for index, val1 in enumerate(emb1.tolist()):
          if index < 7:
            for val in val1:
              self.assertNotEqual(val, 1.0)
          else:
            for val in val1:
              self.assertEqual(val, .0)

  def _RecordFreqTestTemplate(self, optimizer):
    checkpoint_directory = self.get_temp_dir()
    os.environ["TF_RECORD_FREQ"] = "1"
    with ops.device("/cpu:0"):
      emb_var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(emb_var,
            math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = self._CreateOptimizer(optimizer)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, gs)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, train_op])
      sess.run([emb, train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-freqs":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 6)
          self.assertEqual(ckpt_value.tolist()[1], 4)
          self.assertEqual(ckpt_value.tolist()[2], 2)
    os.environ["TF_RECORD_FREQ"] = "0"

  def _RecordVersionTemplate(self, optimizer):
    checkpoint_directory = self.get_temp_dir()
    os.environ["TF_RECORD_VERSION"] = "1"
    with ops.device("/cpu:0"):
      emb_var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(emb_var,
            math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = self._CreateOptimizer(optimizer)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, gs)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, train_op])
      sess.run([emb, train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-versions":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          if "AdagradDecay" in optimizer:
            self.assertEqual(ckpt_value.tolist()[0], 2)
            self.assertEqual(ckpt_value.tolist()[1], 2)
            self.assertEqual(ckpt_value.tolist()[2], 2)
          else:
            self.assertEqual(ckpt_value.tolist()[0], 1)
            self.assertEqual(ckpt_value.tolist()[1], 1)
            self.assertEqual(ckpt_value.tolist()[2], 1)
    os.environ["TF_RECORD_VERSION"] = "0"

  def testSaveVersionWithGlobalStepEviction(self):
    print("testSaveVersionWithGlobalStepEviction")
    checkpoint_directory = self.get_temp_dir()
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
                                          embedding_dim=6,
                                          initializer=init_ops.ones_initializer,
                                          steps_to_live = 5)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([5], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)
    init = variables.global_variables_initializer()
    saver = saver_module.Saver(sharded=True)
    model_path = os.path.join(checkpoint_directory, "model.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      r, _, _ = sess.run([emb, train_op,loss])
      r, _, _ = sess.run([emb, train_op,loss])
      saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-versions":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 1)

  def testFeatureColumnRecordFreqWithPartition(self):
    print("testFeatureColumnRecordFreqWithPartition")
    os.environ["TF_RECORD_FREQ"] = "1"
    checkpoint_directory = self.get_temp_dir()
    columns = feature_column.sparse_column_with_embedding(
                                        column_name="col_emb",
                                        dtype=dtypes.int64,
                                        partition_num=2)
    W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            combiner="mean")
    ids = {}
    ids["col_emb"] = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([0,0,0,1,1,2], dtypes.int64),
                      dense_shape=[6, 1])
    with ops.device("/cpu:0"):
      emb= feature_column_ops.input_from_feature_columns(
             columns_to_tensors=ids, feature_columns=[W])
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, train_op])
      sess.run([emb, train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if "part_0-freqs" in name and name.endswith("-freqs"):
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 6)
          self.assertEqual(ckpt_value.tolist()[1], 2)
        elif "part_1-freqs" in name and name.endswith("-freqs"):
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 4)
    os.environ["TF_RECORD_FREQ"] = "0"

  def testFeatureColumnRecordFreqSGDWithPartition(self):
    print("testFeatureColumnRecordFreqSGDWithPartition")
    os.environ["TF_RECORD_FREQ"] = "1"
    checkpoint_directory = self.get_temp_dir()
    columns = feature_column.sparse_column_with_embedding(
                                        column_name="col_emb",
                                        dtype=dtypes.int64,
                                        partition_num=2)
    W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            combiner="mean")
    ids = {}
    ids["col_emb"] = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([0,0,0,1,1,2], dtypes.int64),
                      dense_shape=[6, 1])
    with ops.device("/cpu:0"):
      emb= feature_column_ops.input_from_feature_columns(
             columns_to_tensors=ids, feature_columns=[W])
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = gradient_descent.GradientDescentOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, train_op])
      sess.run([emb, train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if "part_0-freqs" in name and name.endswith("-freqs"):
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 6)
          self.assertEqual(ckpt_value.tolist()[1], 2)
        elif "part_1-freqs" in name and name.endswith("-freqs"):
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 4)
    os.environ["TF_RECORD_FREQ"] = "0"

  def testDynamicDimensionEmbeddingVariable(self):
    print("testDynamicDimensionEmbeddingVariable")
    def runTestAdagradDecay(self, var, g):
      if isinstance(var, kv_variable_ops.EmbeddingVariable):
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      else:
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64), blocknums=[2,2,2,2,2,2])
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
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
    with ops.Graph().as_default() as g:
      emb_var = variable_scope.get_embedding_variable("var_1",
            initializer=init_ops.ones_initializer(dtypes.float32),
            embedding_dim = 8,
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      emb1 = runTestAdagradDecay(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var =  variable_scope.get_dynamic_dimension_embedding_variable("var_dist",
                                                                    embedding_block_dimension=4,
                                                                    embedding_block_num=2,
                                                                    initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdagradDecay(self, var, g)

    for i in range(0, 6):
      for j in range(0, 8):
        self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testDynamicEmbeddingVariableForInitFromProto(self):
    print("testDynamicEmbeddingVariableForInitFromProto")
    embedding = variable_scope.get_dynamic_dimension_embedding_variable("var_dist",
                                                                    embedding_block_dimension=4,
                                                                    embedding_block_num=2,
                                                                    initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(embedding, math_ops.cast([0,1,2,5,6,7], dtypes.int64), blocknums=[2,2,2,2,2,2])
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    graph = ops.get_default_graph()
    saver = saver_module.Saver(sharded=True)
    meta_graph_def = saver_module.export_meta_graph(saver_def=saver.as_saver_def())
    ops.reset_default_graph()
    with self.test_session() as sess:
      res = saver_module.import_meta_graph(meta_graph_def)

  def testEmbeddingVariableForInitFromProto(self):
    print("testEmbeddingVariableForInitFromProto")
    embedding = variable_scope.get_embedding_variable("var_dist",
                                          embedding_dim=6,
                                          initializer=init_ops.ones_initializer,
                                          steps_to_live = 4,
                                          partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(embedding, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    graph = ops.get_default_graph()
    saver = saver_module.Saver(sharded=True)
    meta_graph_def = saver_module.export_meta_graph(saver_def=saver.as_saver_def())
    ops.reset_default_graph()
    with self.test_session() as sess:
      res = saver_module.import_meta_graph(meta_graph_def)

  def testEmbeddingVariableForLookupInt64(self):
    print("testEmbeddingVariableForLookupInt64")
    with ops.device("/cpu:0"):
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
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
                                                  embedding_dim = 3,
                                                  key_dtype=dtypes.int32,
                                                  initializer=init_ops.ones_initializer(dtypes.float32),
                                                  partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,-7], dtypes.int32))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = adam.AdamOptimizer(0.01)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([train_op])
      model_path = os.path.join(checkpoint_directory, "model.ckpt")
      save_path = saver.save(sess, model_path, global_step=12345)
      saver.restore(sess, save_path)

  def testEmbeddingVariableForGetShape(self):
    print("testEmbeddingVariableForGetShape")
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = adam.AdamOptimizer(0.01)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    shape = var.total_count()
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([train_op])
      self.assertAllEqual([6, 3], sess.run(shape))

  def testEmbeddingVariableForMultiHashFunction(self):
    print("testEmbeddingVariableForMultiHashFunction")
    operation_list = ['concat', 'add', 'mul']
    for operation in operation_list:
      print("testEmbeddingVariableForMultiHashFunction:" + operation)
      var1 = variable_scope.get_variable("var_1_"+operation, shape=[5,6],
                                      initializer=init_ops.ones_initializer(dtypes.float32),
                                      partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      var2 = variable_scope.get_variable("var_2_"+operation, shape=[3,6],
                                      initializer=init_ops.ones_initializer(dtypes.float32),
                                      partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      ids_Q = math_ops.cast([0//5, 1//5, 2//5 , 4//5, 6//5, 7//5],dtypes.int64)
      ids_R = math_ops.cast([0%3, 1%3, 2%3 , 4%3, 6%3, 7%3],dtypes.int64)
      emb1 =  embedding_ops.embedding_lookup(var1, ids_Q)
      emb2 =  embedding_ops.embedding_lookup(var2, ids_R)
      if operation == 'concat':
        emb = array_ops.concat([emb1, emb2], 1)
      elif operation == 'add':
        emb = math_ops.add(emb1, emb2)
      elif operation == 'mul':
        emb = math_ops.multiply(emb1, emb2)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)

      ids = math_ops.cast([0, 1, 2, 4, 6, 7], dtypes.int64)
      var_multi = variable_scope.get_multihash_variable("var_multi_"+operation,
                                          [[5,6],[3,6]],
                                          complementary_strategy="Q-R",
                                          operation=operation,
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

  def testCategoricalColumnWithEmbeddingVariableFunction(self):
    print("testCategoricalColumnWithEmbeddingVariableFunction")
    operation_list = ['concat', 'add', 'mul']
    for operation in operation_list:
      var1 = variable_scope.get_variable("var_1_"+operation, shape=[5,6],
                                        initializer=init_ops.ones_initializer(dtypes.float32),
                                        partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      var2 = variable_scope.get_variable("var_2_"+operation, shape=[3,6],
                                        initializer=init_ops.ones_initializer(dtypes.float32),
                                        partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2))
      ids_Q = math_ops.cast([0//5, 1//5, 2//5 , 4//5, 6//5, 7//5],dtypes.int64)
      ids_R = math_ops.cast([0%3, 1%3, 2%3 , 4%3, 6%3, 7%3],dtypes.int64)
      emb1 =  embedding_ops.embedding_lookup(var1, ids_Q)
      emb2 =  embedding_ops.embedding_lookup(var2, ids_R)
      if operation == 'concat':
        emb = array_ops.concat([emb1, emb2], 1)
      elif operation == 'add':
        emb = math_ops.add(emb1, emb2)
      elif operation == 'mul':
        emb = math_ops.multiply(emb1, emb2)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)

      ids={}
      col_name = "col_emb_" + operation
      ids[col_name] = sparse_tensor.SparseTensor(indices=[[0],[1],[2],[3],[4],[5]], values=math_ops.cast([0, 1, 2, 4, 6, 7], dtypes.int64), dense_shape=[6])
      columns = feature_column_v2.categorical_column_with_multihash(key = col_name,
                                                                    dims = (5,3),
                                                                    complementary_strategy="Q-R",
                                                                    operation=operation,)
      W = feature_column_v2.embedding_column(categorical_column=columns,
                                             dimension=(6,6),
                                             initializer=init_ops.ones_initializer)
      emb_multi = feature_column_v1.input_layer(ids, [W])
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
        for i in range(ids[col_name].shape.as_list()[0]):
          self.assertAllEqual(val_list[0][i], val_list[1][i])

  def testEmbeddinVariableForPartitionOffset(self):
    print("testEmbeddinVariableForPartitionOffset")
    checkpoint_directory = self.get_temp_dir()
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1", embedding_dim = 3)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0, 1, 1000, 1001, 2, 1002], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([init])
      sess.run(train_op)
      model_path = os.path.join(checkpoint_directory, "model.ckpt")
      saver.save(sess, model_path)

    for name, shape in checkpoint_utils.list_variables(model_path):
      if "partition_offset" in name:
        self.assertEqual(shape[0], 1001)
        part_offset = checkpoint_utils.load_variable(model_path, name)
        self.assertEqual(part_offset[0], 0)
        self.assertEqual(part_offset[1], 2)
        self.assertEqual(part_offset[2], 4)
        for i in range(3, len(part_offset)):
          self.assertEqual(part_offset[i], 6)

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
    with ops.device("/cpu:0"):
      emb= feature_column_ops.input_from_feature_columns(
             columns_to_tensors=ids, feature_columns=[W])
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver(sharded=True)
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

  def testEmbeddingVariableForGlobalStepEviction(self):
    print("testEmbeddingVariableForGlobalStepEviction")
    checkpoint_directory = self.get_temp_dir()
    ckpt_path = os.path.join(checkpoint_directory, "model1.ckpt")
    evict = variables.GlobalStepEvict(steps_to_live=2)
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(evict_option=evict))
    ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
    emb = embedding_ops.embedding_lookup(var, ids)
    gs = training_util.get_or_create_global_step()
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([init])
      sess.run([train_op], feed_dict={ids:[1,2,3]})
      sess.run([train_op], feed_dict={ids:[2,3]})
      sess.run([train_op], feed_dict={ids:[2,3]})
      sess.run([train_op], feed_dict={ids:[2,3]})
      saver.save(sess, ckpt_path)
    for name, shape in checkpoint_utils.list_variables(ckpt_path):
      if name == "var_1-keys":
        right_result = [2, 3]
        self.assertEqual(shape[0], 2)
        keys = checkpoint_utils.load_variable(ckpt_path, name)
        for i in range(shape[0]):
          self.assertEqual(keys[i], right_result[i])
      elif name == "var_1-versions":
        right_result = [3, 3]
        self.assertEqual(shape[0], 2)
        keys = checkpoint_utils.load_variable(ckpt_path, name)
        for i in range(shape[0]):
          self.assertEqual(keys[i], right_result[i])

  def testEmbeddingVariableForL2FeatureEviction(self):
    print("testEmbeddingVariableForL2FeatureEviction")
    checkpoint_directory = self.get_temp_dir()
    evict = variables.L2WeightEvict(l2_weight_threshold=0.9)
    with ops.device("/cpu:0"):
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
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      emb_ori = sess.run([emb, train_op])
      save_path = saver.save(sess, os.path.join(checkpoint_directory, "model1.ckpt"), global_step=12345)
      for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
        if name == "var_1-keys":
          self.assertEqual(shape[0], 2)
          keys = checkpoint_utils.load_variable(checkpoint_directory, name)
          self.assertAllEqual(keys, [0, 1])

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
    with ops.device("/cpu:0"):
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
    columns = feature_column.sparse_column_with_embedding(column_name="col_emb", dtype=dtypes.int64)
    W = feature_column.embedding_column(sparse_id_column=columns,
            dimension=3,
            initializer=init_ops.ones_initializer(dtypes.float32))

    ids={}
    ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,4]], values=math_ops.cast([1,2,3,4,5], dtypes.int64), dense_shape=[5, 5])

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
      with ops.device("/cpu:0"):
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
        self.assertEqual(val, .0)
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
        self.assertEqual(val, .0)
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
        self.assertEqual(val, .0)
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
        self.assertEqual(val, .0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)

  def testEmbeddingVariableForAdagradDecayFilter(self):
    print("testEmbeddingVariableForAdagradDecayFilter")
    self._CounterFilterTestTemplate("AdagradDecay")

  def testEmbeddingVariableForAdagradFilter(self):
    print("testEmbeddingVariableForAdagradFilter")
    self._CounterFilterTestTemplate("Adagrad")

  def testEmbeddingVariableForFtrlFilter(self):
    print("testEmbeddingVariableForFtrlFilter")
    self._CounterFilterTestTemplate("FTRL")

  def testEmbeddingVariableForAdamAsyncFilter(self):
    print("testEmbeddingVariableForAdamAsynsFilter")
    self._CounterFilterTestTemplate("AdamAsync")

  def testEmbeddingVariableForGradientDescentFilter(self):
    print("testEmbeddingVariableForGradientDescentFilter")
    self._CounterFilterTestTemplate("GradientDescent")

  def testEmbeddingVariableForAdagradDecayV2Filter(self):
    print("testEmbeddingVariableForAdagradDecayV2Filter")
    self._CounterFilterTestTemplate("AdagradDecayV2")

  def testEmbeddingVariableForAdamFilter(self):
    print("testEmbeddingVariableForAdamFilter")
    self._CounterFilterTestTemplate("Adam")

  def testEmbeddingVariableForAdamWFilter(self):
    print("testEmbeddingVariableForAdamWFilter")
    self._CounterFilterTestTemplate("AdamW")

  def testEmbeddingVariableForGradientDescent(self):
    print("testEmbeddingVariableForGradientDescent")
    self._OpitmizerTestTemplate("GradientDescent")

  def testEmbeddingVariableForAdagrad(self):
    print("testEmbeddingVariableForAdagrad")
    self._OpitmizerTestTemplate("Adagrad")

  def testEmbeddingVariableForAdagradDecay(self):
    print("testEmbeddingVariableForAdagradDecay")
    self._OpitmizerTestTemplate("AdagradDecay")

  def testEmbeddingVariableForAdagradDecayV2(self):
    print("testEmbeddingVariableForAdagradDecayV2")
    self._OpitmizerTestTemplate("AdagradDecayV2")

  def testEmbeddingVariableForAdam(self):
    print("testEmbeddingVariableForAdam")
    self._OpitmizerTestTemplate("Adam")

  def testEmbeddingVariableForAdamAsync(self):
    print("testEmbeddingVariableForAdamAsync")
    self._OpitmizerTestTemplate("AdamAsync")

  def testEmbeddingVariableForAdagradDecayRecrodFreq(self):
    print("testEmbeddingVariableForAdagradDecayRecrodFreq")
    self._RecordFreqTestTemplate("AdagradDecay")

  def testEmbeddingVariableForAdagradRecordFreq(self):
    print("testEmbeddingVariableForAdagradRecordFreq")
    self._RecordFreqTestTemplate("Adagrad")

  def testEmbeddingVariableForFtrlRecrodFreq(self):
    print("testEmbeddingVariableForFtrlRecrodFreq")
    self._RecordFreqTestTemplate("FTRL")

  def testEmbeddingVariableForAdamAsyncRecrodFreq(self):
    print("testEmbeddingVariableForAdamAsyncRecrodFreq")
    self._RecordFreqTestTemplate("AdamAsync")

  def testEmbeddingVariableForGradientDescentRecrodFreq(self):
    print("testEmbeddingVariableForGradientDescentRecrodFreq")
    self._RecordFreqTestTemplate("GradientDescent")

  def testEmbeddingVariableForAdagradDecayV2RecrodFreq(self):
    print("testEmbeddingVariableForAdagradDecayV2RecrodFreq")
    self._RecordFreqTestTemplate("AdagradDecayV2")

  def testEmbeddingVariableForAdamRecrodFreq(self):
    print("testEmbeddingVariableForAdamRecrodFreq")
    self._RecordFreqTestTemplate("Adam")

  def testEmbeddingVariableForAdamWRecrodFreq(self):
    print("testEmbeddingVariableForAdamWRecrodFreq")
    self._RecordFreqTestTemplate("AdamW")

  def testEmbeddingVariableForAdagradDecayRecrodVersion(self):
    print("testEmbeddingVariableForAdagradDecayRecrodVersion")
    self._RecordFreqTestTemplate("AdagradDecay")

  def testEmbeddingVariableForAdagradRecordVersion(self):
    print("testEmbeddingVariableForAdagradRecordVersion")
    self._RecordFreqTestTemplate("Adagrad")

  def testEmbeddingVariableForAdamAsyncRecrodVersion(self):
    print("testEmbeddingVariableForAdamAsyncRecrodVersion")
    self._RecordFreqTestTemplate("AdamAsync")

  def testEmbeddingVariableForGradientDescentRecrodVersion(self):
    print("testEmbeddingVariableForGradientDescentRecrodVersion")
    self._RecordFreqTestTemplate("GradientDescent")

  def testEmbeddingVariableForAdagradDecayV2RecrodVersion(self):
    print("testEmbeddingVariableForAdagradDecayV2RecrodVersion")
    self._RecordFreqTestTemplate("AdagradDecayV2")

  def testEmbeddingVariableForAdamRecrodVersion(self):
    print("testEmbeddingVariableForAdamRecrodVersion")
    self._RecordFreqTestTemplate("Adam")

  def testEmbeddingVariableForAdamWRecrodVersion(self):
    print("testEmbeddingVariableForAdamWRecrodVersion")
    self._RecordFreqTestTemplate("AdamW")

  def testEmbeddingVariableWeightedCategoricalColumn(self):
    print("testEmbeddingVariableWeightedCategoricalColumn")
    with ops.device('/cpu:0'):
      def runTestColumn(W):
        ids={}
        ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                            values=math_ops.cast([1,3,2,3,4,5,3], dtypes.int64), dense_shape=[5, 5])    
        ids['weight'] = [[2.0],[5.0],[4.0],[8.0],[3.0],[1.0],[2.5]]

        emb = feature_column_v1.input_layer(ids, [W])
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
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r

      columns = feature_column_v2.categorical_column_with_identity("col_emb", num_buckets=6)
      emb_columns = feature_column_v2.categorical_column_with_embedding("col_emb", dtype=dtypes.int64)

      columns = feature_column_v2.weighted_categorical_column(columns, 'weight')
      emb_columns = feature_column_v2.weighted_categorical_column(emb_columns, 'weight')

      W = feature_column_v2.embedding_column(categorical_column=columns, dimension=3,
                              initializer=init_ops.ones_initializer(dtypes.float32))
      emb_W = feature_column_v2.embedding_column(categorical_column=emb_columns, dimension=3,
                              initializer=init_ops.ones_initializer(dtypes.float32))

      emb1 = runTestColumn(W)
      emb2 = runTestColumn(emb_W)

      for i in range(0, 5):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableSequenceCategoricalColumn(self):
    print("testEmbeddingVariableSequenceCategoricalColumn")
    with ops.device('/cpu:0'):
      def runTestColumn(W):
        ids={}
        ids["col_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[0,1],[1,1],[2,2],[3,3],[4,3],[4,4]], \
                            values=math_ops.cast([1,3,2,3,4,5,3], dtypes.int64), dense_shape=[5, 5])

        from tensorflow.contrib.feature_column import sequence_input_layer
        emb, _ = sequence_input_layer(ids, [W])
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
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          r, _, _ = sess.run([emb, train_op,loss])
          return r

      from tensorflow.python.feature_column import sequence_feature_column
      columns = sequence_feature_column.sequence_categorical_column_with_identity(key="col_emb", num_buckets=6)
      emb_columns = sequence_feature_column.sequence_categorical_column_with_embedding(key="col_emb", dtype=dtypes.int64)

      W = feature_column_v2.embedding_column(categorical_column=columns, dimension=3,
                              initializer=init_ops.ones_initializer(dtypes.float32))
      emb_W = feature_column_v2.embedding_column(categorical_column=emb_columns, dimension=3,
                                  initializer=init_ops.ones_initializer(dtypes.float32))

      emb1 = runTestColumn(W)
      emb2 = runTestColumn(emb_W)

      for i in range(0, 5):
        for j in range(0, 3):
          self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

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
    with ops.device("/cpu:0"):
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
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
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
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
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
    with ops.device("/cpu:0"):
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

  def testEVInitializerWithKeyFetch(self):
    print("testEVInitializerWithKeyFetch")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_variable("var", shape=[8,3],
                                        initializer=init_ops.glorot_uniform_initializer(seed = 1))
      init_opt = variables.InitializerOption(initializer=init_ops.glorot_uniform_initializer(seed = 1),
                                         default_value_dim=8)
      ev_option = variables.EmbeddingVariableOption(init_option=init_opt)
      emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim=3,
                                                       ev_option=ev_option)
      var_emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,3,4,5,6,7], dtypes.int64))
      emb_emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([0,1,2,5,6,7,8,9,10], dtypes.int64))
      fun = math_ops.multiply(emb_emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1 = sess.run(var_emb)
        emb2, _ = sess.run([emb_emb, train_op])
        self.assertListEqual(emb1.tolist()[0], emb2.tolist()[0])
        self.assertListEqual(emb1.tolist()[1], emb2.tolist()[1])
        self.assertListEqual(emb1.tolist()[2], emb2.tolist()[2])
        self.assertListEqual(emb1.tolist()[5], emb2.tolist()[3])
        self.assertListEqual(emb1.tolist()[6], emb2.tolist()[4])
        self.assertListEqual(emb1.tolist()[7], emb2.tolist()[5])
        self.assertListEqual(emb1.tolist()[0], emb2.tolist()[6])
        self.assertListEqual(emb1.tolist()[1], emb2.tolist()[7])
        self.assertListEqual(emb1.tolist()[2], emb2.tolist()[8])

  def testEVInitializerWithCounterFeatureFilter(self):
    print("testEVInitializerWithCounterFeatureFilter")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_variable("var", shape=[8,3],
                                        initializer=init_ops.glorot_uniform_initializer(seed = 1))
      counter_filter_option=variables.CounterFilter(filter_freq=3)
      init_opt = variables.InitializerOption(initializer=init_ops.glorot_uniform_initializer(seed = 1),
                                         default_value_dim=8)
      ev_option = variables.EmbeddingVariableOption(init_option=init_opt, filter_option=counter_filter_option)
      emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim=3,
                                                       ev_option=ev_option)
      var_emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,3,4,5,6,7], dtypes.int64))
      emb_emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([3], dtypes.int64))
      fun = math_ops.multiply(emb_emb, 0.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1 = np.zeros([8,3])
        emb2, _ = sess.run([emb_emb, train_op])
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2, _ = sess.run([emb_emb, train_op])
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2, _ = sess.run([emb_emb, train_op])
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb1 = sess.run(var_emb)
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])

  def testEVInitializerWithBloomFeatureFilter(self):
    print("testEVInitializerWithBloomFeatureFilter")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_variable("var", shape=[8,3],
                                        initializer=init_ops.glorot_uniform_initializer(seed = 1))
      bloom_filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 10,
                                      false_positive_probability = 0.01)
      init_opt = variables.InitializerOption(initializer=init_ops.glorot_uniform_initializer(seed = 1),
                                         default_value_dim=8)
      ev_option = variables.EmbeddingVariableOption(init_option=init_opt, filter_option=bloom_filter_option)
      emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim=3,
                                                       ev_option=ev_option)
      var_emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,3,4,5,6,7], dtypes.int64))
      emb_emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([3], dtypes.int64))
      fun = math_ops.multiply(emb_emb, 0.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1 = np.zeros([8,3])
        emb2, _ = sess.run([emb_emb, train_op])
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2, _ = sess.run([emb_emb, train_op])
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb2, _ = sess.run([emb_emb, train_op])
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])
        emb1 = sess.run(var_emb)
        emb2 = sess.run(emb_emb)
        self.assertListEqual(emb1.tolist()[3], emb2.tolist()[0])

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
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        r, _, _ = sess.run([emb, train_op, loss])
        return r
    print("testEVInitializer")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      init = variables.InitializerOption(default_value_dim=8192)
      ev_option = variables.EmbeddingVariableOption(init_option = init)
      emb_var = variable_scope.get_embedding_variable("emb_var", embedding_dim = 6,
                                        initializer=init_ops.glorot_uniform_initializer(seed = 3),
                                        ev_option = ev_option)

      var = variable_scope.get_variable("var", shape=[8192, 6],
                                   initializer=init_ops.glorot_uniform_initializer(seed = 3))
      emb1 = runTest(self, emb_var, g)
      emb2 = runTest(self, var, g)

      for i in range(0, 6):
        for j in range(0, 6):
          self.assertAllCloseAccordingToType(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForDRAM(self):
    print("testEmbeddingVariableForDRAM")
    def runTestAdagrad(self, var, g):
      search_list=[]
      for i in range(0, 1024 * 2):
        search_list.append(i)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast(search_list, dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      #print(ops.get_default_graph().as_graph_def())
      #config = config_pb3.ConfigProto(log_device_placement=True)
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        r = sess.run([emb])
        r, _, _ = sess.run([emb, train_op,loss])
        r, _, _ = sess.run([emb, train_op,loss])
        return r

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 128,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM)))
      var = variable_scope.get_variable("var_2", shape=[1024 * 512, 3], initializer=init_ops.ones_initializer(dtypes.float32))
      time_start = time.time()
      emb1 = runTestAdagrad(self, emb_var, g)
      print(emb1)
      time_end = time.time()
      time_c = time_end - time_start
      print('time cost', time_c, 's')
      #emb2 = runTestAdagrad(self, var, g)
      #print(emb2)

      #for i in range(0, 6):
        #for j in range(0, 3):
          #self.assertEqual(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForSaveFreq(self):
    db_directory = self.get_temp_dir()
    checkpoint_directory = self.get_temp_dir()
    emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                                                                                                 storage_path=db_directory)))
    emb = embedding_ops.embedding_lookup(emb_var, math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay_v2.AdagradDecayOptimizerV2(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    saver = saver_module.Saver(sharded=True)
    model_path = os.path.join(checkpoint_directory, "model.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      r, _, _ = sess.run([emb, train_op,loss])
      r, _, _ = sess.run([emb, train_op,loss])
      saver.save(sess, model_path)
      r, _ = sess.run([emb, loss])
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-freqs":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 6)
          self.assertEqual(ckpt_value.tolist()[1], 4)
          self.assertEqual(ckpt_value.tolist()[2], 2)


  def testEmbeddingVariableForL2FeatureEvictionDRAM(self):
    print("testEmbeddingVariableForL2FeatureEvictionDRAM")
    checkpoint_directory = self.get_temp_dir()
    db_directory = self.get_temp_dir()
    evict = variables.L2WeightEvict(l2_weight_threshold=0.9)
    storage_option = variables.StorageOption(storage_type=config_pb2.StorageType.DRAM)
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(evict_option=evict, storage_option=storage_option))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,0,0,1,1,2], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = ftrl.FtrlOptimizer(0.1, l1_regularization_strength=2.0, l2_regularization_strength=0.00001)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    saver = saver_module.Saver(sharded=True)
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

  def testEmbeddingVariableForDRAMAndLEVELDB(self):
    print("testEmbeddingVariableForDRAMAndLEVELDB")
    def runTestAdagrad(self, var, g):
      #ids = array_ops.placeholder(dtypes.int64, name="ids")
      #emb = embedding_ops.embedding_lookup(var, ids)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 2, 3, 4, 5, 6, 7, 8, 9], dtypes.int64))
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
        for i in xrange(60):
          r, _, _ = sess.run([emb, train_op, loss])
        return r

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      db_directory = self.get_temp_dir()
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1),
            steps_to_live=5,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                                                                                                 storage_path=db_directory,
                                                                                                 storage_size=[4096])))
      emb1 = runTestAdagrad(self, emb_var, g)

    with ops.Graph().as_default() as g:
      var = variable_scope.get_variable("var_2", shape=[100, 30], initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdagrad(self, var, g)

      for i in range(0, 9):
        for j in range(0, 30):
          self.assertAllCloseAccordingToType(emb1.tolist()[i][j], emb2.tolist()[i][j])

  def testEmbeddingVariableForDRAMAndSSD(self):
    print("testEmbeddingVariableForDRAMAndSSD")
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
        r1 = sess.run(emb, {ids:[1,2,4,5]})
        r2 = sess.run(emb, {ids:[3]})
        r = r1.tolist() + r2.tolist()
        return r

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      db_directory = self.get_temp_dir()
      os.environ["TF_SSDHASH_ASYNC_COMPACTION"]="0"
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.DRAM_SSDHASH,
                        storage_path=db_directory,
                        storage_size=[1024])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = ev_option)
      emb1 = runTestAdagrad(self, emb_var, g)

    with ops.Graph().as_default() as g, ops.device("/cpu:0"):
      var = variable_scope.get_variable("var_2",
                shape=[100, 30],
                initializer=init_ops.ones_initializer(dtypes.float32))
      emb2 = runTestAdagrad(self, var, g)

    del os.environ["TF_SSDHASH_ASYNC_COMPACTION"]
    for i in range(0, 5):
      for j in range(0, 30):
        self.assertAllCloseAccordingToType(emb1[i][j], emb2[i][j])

  def testEmbeddingVariableForDRAMAndSSDSaveCkpt(self):
    print("testEmbeddingVariableForDRAMAndSSDSaveCkpt")
    checkpoint_directory = self.get_temp_dir()
    def readSsdRecord(model_path):
      ssd_record_path = model_path + "-var_1-ssd_record"
      for name, shape in checkpoint_utils.list_variables(ssd_record_path):
        ckpt_value = checkpoint_utils.load_variable(ssd_record_path, name)
        if name == "keys":
          self.assertAllEqual(ckpt_value, [3, 4 ,5])
        if name == "keys_file_id":
          self.assertAllEqual(ckpt_value, [0, 0 ,0])
        if name == "keys_offset":
          self.assertAllEqual(ckpt_value, [0, 272 ,544])
        if name == "files":
          self.assertAllEqual(ckpt_value, [0])
        if name == "invalid_record_count":
          self.assertAllEqual(ckpt_value, [0])
        if name == "record_count":
          self.assertAllEqual(ckpt_value, [3])

    def runTestAdagrad(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, global_step=gs)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
      model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
      tires = kv_variable_ops.lookup_tier(emb_var,
                  math_ops.cast([1,2,3,4,5,6,7], dtypes.int64))
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[1,2,3]})
        sess.run([train_op], {ids:[1,2,4]})
        sess.run([train_op], {ids:[1,2,2]})
        sess.run([train_op], {ids:[1,2,5]})
        sess.run([train_op], {ids:[1,2,6]})
        sess.run([train_op], {ids:[1,2,7]})
        result = sess.run(tires)
        for i in range(0, 7):
          if i in range(2, 5):
            self.assertEqual(result[i], 1)
          else:
            self.assertEqual(result[i], 0)
        saver.save(sess, model_path)
        readSsdRecord(model_path)

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      os.environ["TF_SSDHASH_ASYNC_COMPACTION"]="0"
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.DRAM_SSDHASH,
                        storage_path=checkpoint_directory,
                        storage_size=[1024])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 30,
            initializer=init_ops.ones_initializer(dtypes.float32),
            steps_to_live=5,
            ev_option = ev_option)
      runTestAdagrad(self, emb_var, g)

    del os.environ["TF_SSDHASH_ASYNC_COMPACTION"]

  def testEmbeddingVariableForDramAndLevelDBSaveCkpt(self):
    print("testEmbeddingVariableForDramAndLevelDBSaveCkpt")
    checkpoint_directory = self.get_temp_dir()
    def runTestAdagrad(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, global_step=gs)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
      model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
      tires = kv_variable_ops.lookup_tier(emb_var,
                  math_ops.cast([0,1,2,3,4,5,6,7,8,9,10,11], dtypes.int64))
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[0,1,2,3,4,5]})
        sess.run([train_op], {ids:[6,7,8,9,10,11]})
        sess.run([train_op], {ids:[0,1,2,3,4,5]})
        result = sess.run(tires)
        for i in range(0, 12):
          if i in range(0, 6):
            self.assertEqual(result[i], 0)
          else:
            self.assertEqual(result[i], 1)
        saver.save(sess, model_path)
        for name, shape in checkpoint_utils.list_variables(model_path):
          if name == "var_1-keys" or name == "var_1/Adagrad-keys":
            self.assertEqual(shape[0], 12)
            keys = checkpoint_utils.load_variable(model_path, name)
            self.assertAllEqual(np.array([0,1,2,3,4,5,6,7,8,9,10,11]), keys)
          if name == "var_1-freqs" or name == "var_1/Adagrad-freqs":
            freqs = checkpoint_utils.load_variable(model_path, name)
            self.assertAllEqual(np.array([2,2,2,2,2,2,1,1,1,1,1,1]), freqs)
          if name == "var_1/Adagrad-values":
            values = checkpoint_utils.load_variable(model_path, name)
            for i in range(0, shape[0]):
              for j in range(0, shape[1]):
                if i < 6:
                  self.assertAlmostEqual(values[i][j], 8.1, delta=1e-05)
                else:
                  self.assertAlmostEqual(values[i][j], 4.1, delta=1e-05)
          if name == "var_1-values":
            values = checkpoint_utils.load_variable(model_path, name)
            for i in range(0, shape[0]):
              for j in range(0, shape[1]):
                if i < 6:
                  self.assertAlmostEqual(values[i][j], 0.8309542, delta=1e-05)
                else:
                  self.assertAlmostEqual(values[i][j], 0.90122706, delta=1e-05)

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                        storage_path = checkpoint_directory,
                        storage_size=[1024 * 6])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 128,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = ev_option)
      runTestAdagrad(self, emb_var, g)

  @test_util.run_gpu_only
  def testEmbeddingVariableForHBMandDRAMSaveCkpt(self):
    print("testEmbeddingVariableForHBMandDRAMSaveCkpt")
    checkpoint_directory = self.get_temp_dir()
    def runTestAdagrad(self, var, g):
      ids = array_ops.placeholder(dtypes.int64, name="ids")
      emb = embedding_ops.embedding_lookup(var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, global_step=gs)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
      model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
      tires = kv_variable_ops.lookup_tier(emb_var,
                  math_ops.cast([0,1,2,3,4,5,6,7,8,9,10,11], dtypes.int64))
      with self.test_session(graph=g) as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run([train_op], {ids:[0,1,2,3,4,5]})
        sess.run([train_op], {ids:[6,7,8,9,10,11]})
        sess.run([train_op], {ids:[0,1,2,3,4,5]})
        result = sess.run(tires)
        for i in range(0, 12):
          if i in range(0, 6):
            self.assertEqual(result[i], 0)
          else:
            self.assertEqual(result[i], 1)
        saver.save(sess, model_path)
        for name, shape in checkpoint_utils.list_variables(model_path):
          if name == "var_1-keys" or name == "var_1/Adagrad-keys":
            self.assertEqual(shape[0], 12)
            keys = checkpoint_utils.load_variable(model_path, name)
            self.assertAllEqual(np.array([0,1,2,3,4,5,6,7,8,9,10,11]), keys)    
          if name == "var_1-freqs" or name == "var_1/Adagrad-freqs":
            freqs = checkpoint_utils.load_variable(model_path, name)
            self.assertAllEqual(np.array([2,2,2,2,2,2,1,1,1,1,1,1]), freqs)
          if name == "var_1/Adagrad-values":
            values = checkpoint_utils.load_variable(model_path, name)
            for i in range(0, shape[0]):
              for j in range(0, shape[1]):
                if i < 6:
                  self.assertAlmostEqual(values[i][j], 8.1, delta=1e-05)
                else:
                  self.assertAlmostEqual(values[i][j], 4.1, delta=1e-05)
          if name == "var_1-values":
            values = checkpoint_utils.load_variable(model_path, name)
            for i in range(0, shape[0]):
              for j in range(0, shape[1]):
                if i < 6:
                  self.assertAlmostEqual(values[i][j], 0.8309542, delta=1e-05)
                else:
                  self.assertAlmostEqual(values[i][j], 0.90122706, delta=1e-05)

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.HBM_DRAM,
                        storage_size=[1024 * 6])
      ev_option = variables.EmbeddingVariableOption(
                                storage_option=storage_option)
      emb_var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 128,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = ev_option)
      runTestAdagrad(self, emb_var, g)
  
  def testEmbeddingVariableForRecordFreq(self):
    print("testEmbeddingVariableForRecordFreq")
    checkpoint_directory = self.get_temp_dir()
    os.environ["TF_RECORD_FREQ"] = "1"
    os.environ["TF_RECORD_VERSION"] = "1"
    with ops.device("/cpu:0"):
      emb_var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(emb_var,
            math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, gs)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, train_op])
      sess.run([emb, train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-freqs":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 6)
          self.assertEqual(ckpt_value.tolist()[1], 4)
          self.assertEqual(ckpt_value.tolist()[2], 2)
        if name == "var_1-versions":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 1)
          self.assertEqual(ckpt_value.tolist()[1], 1)
          self.assertEqual(ckpt_value.tolist()[2], 1)
    os.environ["TF_RECORD_FREQ"] = "0"
    os.environ["TF_RECORD_VERSION"] = "0"

  def testEmbeddingVariableForRecordFreqWithCounterFilter(self):
    print("testEmbeddingVariableForRecordFreqWithCounterFilter")
    checkpoint_directory = self.get_temp_dir()
    os.environ["TF_RECORD_FREQ"] = "1"
    os.environ["TF_RECORD_VERSION"] = "1"
    with ops.device("/cpu:0"):
      emb_var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)))
    emb = embedding_ops.embedding_lookup(emb_var,  math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, gs)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, train_op])
      sess.run([emb, train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-freqs":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 6)
          self.assertEqual(ckpt_value.tolist()[1], 4)
        if name == "var_1-versions":
          ckpt_value = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(ckpt_value.tolist()[0], 1)
          self.assertEqual(ckpt_value.tolist()[1], 1)
    os.environ["TF_RECORD_FREQ"] = "0"
    os.environ["TF_RECORD_VERSION"] = "0"

  def testEmbeddingVariableForDefaultValueNoPermission(self):
    print("testEmbeddingVariableForDefaultValueNoPermission")
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              ev_option = variables.EmbeddingVariableOption(
                            filter_option=variables.CounterFilter(filter_freq=3),
                            init_option=variables.InitializerOption(
                              initializer=init_ops.zeros_initializer(dtypes.float32),
                              default_value_no_permission=.2)),
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
      emb1, _ = sess.run([emb, train_op])
      emb1, _ = sess.run([emb, train_op])
      for val in emb1.tolist()[0]:
        self.assertAlmostEqual(val, .2, delta=1e-05)
      emb1, _ = sess.run([emb, train_op])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, .2)

  def testEmbeddingVariableForGetFrequencyAndVersion(self):
    print("testEmbeddingVariableForGetFrequencyAndVersion")
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(
                filter_option=variables.CounterFilter(filter_freq=3),
                evict_option=variables.GlobalStepEvict(steps_to_live=2))
              )
    shape=var.get_dynamic_shape()
    frequency=var.get_frequency(math_ops.cast([1,2,3,4,5,6,7], dtypes.int64))
    version=var.get_version(math_ops.cast([1,2,3,4,5,6,7], dtypes.int64))
    ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
    emb = embedding_ops.embedding_lookup(var, ids)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, gs)
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, train_op, loss], feed_dict={'ids:0': [1,2,3]})
      sess.run([emb, train_op, loss], feed_dict={'ids:0': [1,3,5]})
      sess.run([emb, train_op, loss], feed_dict={'ids:0': [1,5,7]})
      s, f, v = sess.run([shape, frequency, version])
      self.assertAllEqual(np.array([5,3]), s)
      self.assertAllEqual(np.array([3,1,2,0,2,0,1]), f)
      self.assertAllEqual(np.array([2,0,1,-1,2,-1,2]), v)

  def testEmbeddingVariableForInference(self):
    print("testEmbeddingVariableForInference")
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
                embedding_dim = 3,
                initializer=init_ops.ones_initializer(dtypes.float32))
      shape=var.get_dynamic_shape()
      ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
      emb = embedding_ops.embedding_lookup(var, ids)
    # modify graph for infer
    # emb.op.inputs[0].op.inputs[0].op._set_attr("is_inference", attr_value_pb2.AttrValue(b=True))
    # set environment
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([init])
      sess.run([emb, loss], feed_dict={'ids:0': [1,2,3]})
      sess.run([emb, loss], feed_dict={'ids:0': [1,3,5]})
      sess.run([emb, loss], feed_dict={'ids:0': [1,5,7]})
      s = sess.run(shape)
      self.assertAllEqual(np.array([0,3]), s)

  def testEmbeddingVariableForLookupTier(self):
    print("testEmbeddingVariableForLookupTier")
    os.environ["TF_SSDHASH_ASYNC_COMPACTION"]="0"
    os.environ["TF_MULTI_TIER_EV_EVICTION_THREADS"]="2"
    db_directory = self.get_temp_dir()
    storage_opt = variables.StorageOption(
                          storage_type=config_pb2.StorageType.DRAM_SSDHASH,
                          storage_path=db_directory,
                          storage_size=[512])
    ev_option = variables.EmbeddingVariableOption(storage_option=storage_opt)
    partitioner = partitioned_variables.fixed_size_partitioner(num_shards=2)
    with ops.device('/cpu:0'):
      emb_var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 30,
          initializer=init_ops.ones_initializer(dtypes.float32),
          partitioner=partitioner,
          ev_option = ev_option)
      ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
      emb = embedding_ops.embedding_lookup(emb_var, ids)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, gs)
      init = variables.global_variables_initializer()
      tires = kv_variable_ops.lookup_tier(emb_var,
                  math_ops.cast([1,2,3,4,5,6], dtypes.int64))

    with self.test_session() as sess:
      sess.run([init])
      sess.run(train_op, {ids:[1,2,3]})
      sess.run(train_op, {ids:[1,2,4]})
      sess.run(train_op, {ids:[1,2,2]})
      sess.run(train_op, {ids:[1,2,5]})
      result = sess.run(tires)
      del os.environ["TF_SSDHASH_ASYNC_COMPACTION"]
      del os.environ["TF_MULTI_TIER_EV_EVICTION_THREADS"]
      for i in range(0, 6):
        if i == 2:
          self.assertEqual(result[i], 1)
        elif i == 5:
          self.assertEqual(result[i], -1)
        else:
          self.assertEqual(result[i], 0)
      sess.run(emb, {ids:[3]})
      result = sess.run(tires)
      for i in range(0, 5):
        self.assertEqual(result[i], 0)

  @test_util.run_gpu_only
  def testEmbeddingVariableForHBMandDRAM(self):
    print("testEmbeddingVariableForHBMandDRAM")
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
        sess.run([train_op], {ids:[4, 4]})
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

  @test_util.run_gpu_only
  def testEmbeddingVariableForHBMDRAMAndSSD(self):
    db_directory = self.get_temp_dir()
    print("testEmbeddingVariableForHBMDRAMAndSSD")
    os.environ["TF_SSDHASH_ASYNC_COMPACTION"]="0"
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
        sess.run([train_op], {ids:[1,2,6]})

        if isinstance(var, kv_variable_ops.EmbeddingVariable):
          result = sess.run(tires)
          for i in range(0, 6):
            if i == 2:
              self.assertEqual(result[i], 2)
            elif i == 3:
              self.assertEqual(result[i], 1)
            else:
              self.assertEqual(result[i], 0)

        r1 = sess.run(emb, {ids:[1,2,5,6]})
        r2 = sess.run(emb, {ids:[4, 4]})
        r3 = sess.run(emb, {ids:[3, 3]})
        r = r1.tolist() + r2.tolist() + r3.tolist()
        return r

    with ops.Graph().as_default() as g, ops.device('/gpu:0'):
      storage_option = variables.StorageOption(
                        storage_type=config_pb2.StorageType.HBM_DRAM_SSDHASH,
                        storage_path=db_directory,
                        storage_size=[1024, 256])
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

  def testEmbeddingVariableForContirbFeatureColumnWithPartitionNum(self):
    print("testEmbeddingVariableForContirbFeatureColumnWithPartitionNum")
    checkpoint_directory = self.get_temp_dir()
    evict = variables.L2WeightEvict(l2_weight_threshold=0.9)
    columns = feature_column.sparse_column_with_embedding(
                                        column_name="col_emb",
                                        dtype=dtypes.int64,
                                        partition_num = 4)
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
    saver = saver_module.Saver(sharded=True)

  def testSaveV3(self):
    print("testSaveV3")
    with ops.device("/cpu:0"):
      emb_var = variable_scope.get_embedding_variable("emb_var", 10)
      var = variable_scope.get_variable("var", [10, 10])
    emb1 = embedding_ops.embedding_lookup(emb_var, math_ops.cast([1,2,3], dtypes.int64))
    emb2 = embedding_ops.embedding_lookup(var, math_ops.cast([1,2,3], dtypes.int64))
    emb = emb1 + emb2
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)
    init = variables.global_variables_initializer()
    saver = saver = saver_module.Saver(sharded=True)
    checkpoint_directory = self.get_temp_dir()
    model_path = os.path.join(checkpoint_directory, "model.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      saver.save(sess, model_path)
      sess.run([train_op])
      sess.run([train_op])
      saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        ckpt_value = checkpoint_utils.load_variable(model_path, name)
        print(name, shape, ckpt_value)
    with self.test_session() as sess:
      saver.restore(sess, model_path)

  def testEmbeddingVariableForNotSaveUnfilterFeature(self):
    print("testEmbeddingVariableForNotSaveUnfilterFeature")
    checkpoint_directory = self.get_temp_dir()
    os.environ["TF_EV_SAVE_FILTERED_FEATURES"] = "False"
    with ops.device("/cpu:0"):
      emb_var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)))
    emb = embedding_ops.embedding_lookup(emb_var,  math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, gs)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-keys":
          keys = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(1, len(keys))
          self.assertEqual(1, keys[0])
        if name == "var_1-keys_filtered" or \
           name == "var_1-versions_filtered" or \
           name == "var_1-freqs_filtered":
          self.assertEqual(0, shape[0])
    del os.environ["TF_EV_SAVE_FILTERED_FEATURES"]

  def testEmbeddingVariableForSaveUnfilterFeature(self):
    checkpoint_directory = self.get_temp_dir()
    with ops.device("/cpu:0"):
      emb_var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)))
    emb = embedding_ops.embedding_lookup(emb_var,  math_ops.cast([1, 1, 1, 2, 2, 3], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, gs)
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    model_path = os.path.join(checkpoint_directory,
                              "model1.ckpt")
    with self.test_session() as sess:
      sess.run([init])
      sess.run([train_op])
      save_path = saver.save(sess, model_path)
      for name, shape in checkpoint_utils.list_variables(model_path):
        if name == "var_1-keys":
          keys = checkpoint_utils.load_variable(model_path, name)
          self.assertEqual(1, len(keys))
          self.assertEqual(1, keys[0])
        if name == "var_1-keys_filtered" or \
           name == "var_1-freqs_filtered":
          self.assertEqual(2, shape[0])
  
  def testEmbeddingVariableForMultiTierInference(self):
    print("testEmbeddingVariableForMultiTierInference")
    checkpoint_directory = self.get_temp_dir()
    os.environ["TF_SSDHASH_ASYNC_COMPACTION"]="0"
    os.environ["TF_RECORD_FREQ"] = "1"
    with ops.Graph().as_default() as g, ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var", embedding_dim=30)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,1,1,2,2,3,4], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, gs)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run([init])
        sess.run(train_op)
        saver.save(sess, os.path.join(checkpoint_directory, "model.ckpt"))
    
    with ops.Graph().as_default() as g, ops.device("/cpu:0"):
      db_directory = self.get_temp_dir()
      storage_opt = variables.StorageOption(
                          storage_type=config_pb2.StorageType.DRAM_SSDHASH,
                          storage_path=db_directory,
                          storage_size=[256])
      ev_option = variables.EmbeddingVariableOption(storage_option=storage_opt)
      emb_var = variable_scope.get_embedding_variable("var",
          embedding_dim = 30,
          initializer=init_ops.ones_initializer(dtypes.float32),
          ev_option = ev_option)
      ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
      emb = embedding_ops.embedding_lookup(emb_var, ids)
      tires = kv_variable_ops.lookup_tier(emb_var,
                  math_ops.cast([1,2,3,4], dtypes.int64))
      saver = saver_module.Saver(sharded=True)
      graph = ops.get_default_graph()
      with self.test_session(graph = graph) as sess:
        saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt"))
        result = sess.run(tires)
        self.assertAllEqual(result, [0, 0, 1, 1])
        sess.run(emb, feed_dict={ids:[3, 3]})
        result = sess.run(tires)
        self.assertAllEqual(result, [0, 1, 0, 1])
        print(result)
    del os.environ["TF_SSDHASH_ASYNC_COMPACTION"]
    del os.environ["TF_RECORD_FREQ"]

  def testEmbeddingVariableForSaveAndRestoreForSingleTier(self):
    print("testEmbeddingVariableForSaveAndRestoreForSingleTier")
    checkpoint_directory = self.get_temp_dir()
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2),
              ev_option = variables.EmbeddingVariableOption(
                  storage_option=variables.StorageOption(
                      storage_type=config_pb2.StorageType.DRAM)))
      
      var_2 = variable_scope.get_embedding_variable("var_2",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(
                  storage_option=variables.StorageOption(
                      storage_type=config_pb2.StorageType.DRAM)))

      emb = embedding_ops.embedding_lookup(var,
                                           math_ops.cast([0,1,2,5,6,7],
                                           dtypes.int64))
      emb_1 = embedding_ops.embedding_lookup(var_2,
                                             math_ops.cast([0,1,2,5,6,7],
                                             dtypes.int64))
      fun = math_ops.multiply(emb, 0.0, name='multiply')
      fun1 = math_ops.multiply(emb_1, 0.0, name='multiply_1')
      loss = math_ops.reduce_sum(fun + fun1, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, gs)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        sess.run(train_op)
        emb_ori = sess.run(emb)
        emb_ori_2 = sess.run(emb_1)
        save_path = saver.save(sess, os.path.join(checkpoint_directory, "model.ckpt"), global_step=12345)
        print(save_path)
        for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
          print('loading... ', name, shape)

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2),
              ev_option = variables.EmbeddingVariableOption(
                  storage_option=variables.StorageOption(
                      storage_type=config_pb2.StorageType.DRAM)))
      
      var_2 = variable_scope.get_embedding_variable("var_2",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(
                  storage_option=variables.StorageOption(
                      storage_type=config_pb2.StorageType.DRAM)))

      emb = embedding_ops.embedding_lookup(var,
                                           math_ops.cast([0,1,2,5,6,7],
                                           dtypes.int64))
      emb_1 = embedding_ops.embedding_lookup(var_2,
                                             math_ops.cast([0,1,2,5,6,7],
                                             dtypes.int64))
      saver = saver_module.Saver([var,var_2], sharded=True)
      graph = ops.get_default_graph()
      with self.test_session(graph=graph) as sess:
        saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt-12345"))
        ret = sess.run(emb)
        ret_1 = sess.run(emb_1)
        self.assertAllEqual(emb_ori, ret)
        self.assertAllEqual(emb_ori_2, ret_1)
    
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(
                  storage_option=variables.StorageOption(
                      storage_type=config_pb2.StorageType.DRAM)))
      
      var_2 = variable_scope.get_embedding_variable("var_2",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=2),
              ev_option = variables.EmbeddingVariableOption(
                  storage_option=variables.StorageOption(
                      storage_type=config_pb2.StorageType.DRAM)))

      emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      emb_1 = embedding_ops.embedding_lookup(var_2, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
      saver = saver_module.Saver([var,var_2], sharded=True)
      graph = ops.get_default_graph()
      with self.test_session(graph=graph) as sess:
        saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt-12345"))
        ret = sess.run(emb)
        ret_1 = sess.run(emb_1)
        self.assertAllEqual(emb_ori, ret)
        self.assertAllEqual(emb_ori_2, ret_1)

  def testEmbeddingVariableSaveAndRestoreForMultiTierWithoutHbm(self):
    print("testEmbeddingVariableSaveAndRestoreForMultiTierWithoutHbm")
    checkpoint_directory = self.get_temp_dir()
    os.environ["TF_EV_RESTORE_CUSTOM_DIM"] = "True"
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
        var = variable_scope.get_embedding_variable("var_1",
                embedding_dim = 3,
                ev_option = variables.EmbeddingVariableOption(
                    storage_option=variables.StorageOption(
                        storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                            storage_path='/tmp/leveldb/')))

        var2 = variable_scope.get_embedding_variable("var_2",
            embedding_dim = 3,
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            ev_option = variables.EmbeddingVariableOption(
                storage_option=variables.StorageOption(
                    storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                        storage_path='/tmp/leveldb/')))
        
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        emb2 = embedding_ops.embedding_lookup(var2, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        fun = math_ops.multiply(emb, 0.0, name='multiply')
        fun1 = math_ops.multiply(emb2, 0.0, name='multiply_1')
        loss = math_ops.reduce_sum(fun + fun1, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v, gs)
        saver = saver_module.Saver(sharded=True)
        init = variables.global_variables_initializer()
        graph = ops.get_default_graph()
        with self.test_session() as sess:
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
          sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
          sess.run([init])
          sess.run(train_op)
          emb_ori = sess.run(emb)
          emb_ori_2 = sess.run(emb2)
          save_path = saver.save(sess, os.path.join(checkpoint_directory, "model.ckpt"), global_step=12345)
          print(save_path)
          for name, shape in checkpoint_utils.list_variables(checkpoint_directory):
            print('loading... ', name, shape)

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
        var = variable_scope.get_embedding_variable("var_1",
                embedding_dim = 6,
                ev_option = variables.EmbeddingVariableOption(
                    storage_option=variables.StorageOption(
                        storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                            storage_path='/tmp/leveldb/')))

        var2 = variable_scope.get_embedding_variable("var_2",
            embedding_dim = 5,
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            ev_option = variables.EmbeddingVariableOption(
                storage_option=variables.StorageOption(
                    storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                        storage_path='/tmp/leveldb/')))
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        emb2 = embedding_ops.embedding_lookup(var2, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        saver = saver_module.Saver([var,var2],sharded=True)
        graph = ops.get_default_graph()
        with self.test_session(graph = graph) as sess:
          saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt-12345"))
          emb_val = sess.run(emb)
          emb_val_2 = sess.run(emb2)
          self.assertAllEqual(emb_ori, emb_val[:,0:3])
          self.assertAllEqual(emb_ori_2, emb_val_2[:,0:3])

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
        var = variable_scope.get_embedding_variable("var_1",
                embedding_dim = 2,
                ev_option = variables.EmbeddingVariableOption(
                    storage_option=variables.StorageOption(
                        storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                            storage_path='/tmp/leveldb/')))                                                         

        var2 = variable_scope.get_embedding_variable("var_2",
            embedding_dim = 2,
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            ev_option = variables.EmbeddingVariableOption(
                storage_option=variables.StorageOption(
                    storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                        storage_path='/tmp/leveldb/')))
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        emb2 = embedding_ops.embedding_lookup(var2, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        saver = saver_module.Saver([var,var2],sharded=True)
        graph = ops.get_default_graph()
        with self.test_session(graph = graph) as sess:
          saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt-12345"))
          emb_val = sess.run(emb)
          emb_val_2 = sess.run(emb2)
          self.assertAllEqual(emb_ori[:,0:2], emb_val)
          self.assertAllEqual(emb_ori_2[:,0:2], emb_val_2)
    del os.environ["TF_EV_RESTORE_CUSTOM_DIM"]

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
        var = variable_scope.get_embedding_variable("var_1",
                embedding_dim = 3,
                ev_option = variables.EmbeddingVariableOption(
                    storage_option=variables.StorageOption(
                        storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                            storage_path='/tmp/leveldb/')))

        var2 = variable_scope.get_embedding_variable("var_2",
            embedding_dim = 3,
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            ev_option = variables.EmbeddingVariableOption(
                storage_option=variables.StorageOption(
                    storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                        storage_path='/tmp/leveldb/')))
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        emb2 = embedding_ops.embedding_lookup(var2, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        saver = saver_module.Saver([var,var2],sharded=True)
        graph = ops.get_default_graph()
        with self.test_session(graph = graph) as sess:
          saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt-12345"))
          emb_val = sess.run(emb)
          emb_val_2 = sess.run(emb2)
          self.assertAllEqual(emb_ori, emb_val)
          self.assertAllEqual(emb_ori_2, emb_val_2)

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
        var = variable_scope.get_embedding_variable("var_1",
                embedding_dim = 3,
                ev_option = variables.EmbeddingVariableOption(
                    storage_option=variables.StorageOption(
                        storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                            storage_path='/tmp/leveldb/')))

        var2 = variable_scope.get_embedding_variable("var_2",
            embedding_dim = 3,
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4),
            ev_option = variables.EmbeddingVariableOption(
                storage_option=variables.StorageOption(
                    storage_type=config_pb2.StorageType.DRAM_LEVELDB,
                        storage_path='/tmp/leveldb/')))
        emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        emb2 = embedding_ops.embedding_lookup(var2, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
        saver = saver_module.Saver([var,var2],sharded=True)
        graph = ops.get_default_graph()
        with self.test_session(graph = graph) as sess:
          saver.restore(sess, os.path.join(checkpoint_directory, "model.ckpt-12345"))
          emb_val = sess.run(emb)
          emb_val_2 = sess.run(emb2)
          self.assertAllEqual(emb_ori, emb_val)
          self.assertAllEqual(emb_ori_2, emb_val_2)

  def testCPUFbjOpt(self):
    print("testCPUFbjOpt")
    os.environ["TF_EMBEDDING_FBJ_OPT"] = "True"
    self._OpitmizerTestTemplate("Adagrad")
    del os.environ["TF_EMBEDDING_FBJ_OPT"]
  

  def testCPUFbjOptWithCounterFilter(self):
    print("testCPUFbjOpt")
    os.environ["TF_EMBEDDING_FBJ_OPT"] = "True"
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
              embedding_dim = 3,
              initializer=init_ops.ones_initializer(dtypes.float32),
              ev_option = variables.EmbeddingVariableOption(filter_option=variables.CounterFilter(filter_freq=3)),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=1))
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,1,1,1,2,2,2,3,3,4], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = self._CreateOptimizer("Adagrad")
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      init = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
        sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
        sess.run([init])
        emb1, top, l = sess.run([emb, train_op, loss])
        emb_list = emb1.tolist()
        emb_right = [[.0, .0, .0],
                     [.0, .0, .0],
                     [1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0],
                     [.0, .0, .0],
                     [.0, .0, .0],
                     [1.0, 1.0, 1.0],
                     [.0, .0, .0],
                     [.0, .0, .0],
                     [.0, .0, .0]]
        
        for i in range(6):
          for j in range(3):
            self.assertAlmostEqual(emb_list[i][j], emb_right[i][j])

        emb1= sess.run(emb)
        emb_right = [[0.90031105, 0.90031105, 0.90031105],
                     [0.90031105, 0.90031105, 0.90031105],
                     [0.90031105, 0.90031105, 0.90031105],
                     [0.90031105, 0.90031105, 0.90031105],
                     [0.90122706, 0.90122706, 0.90122706],
                     [0.90122706, 0.90122706, 0.90122706],
                     [0.90122706, 0.90122706, 0.90122706],
                     [1.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0],
                     [.0, .0, .0]]
        for i in range(6):
          for j in range(3):
            self.assertAlmostEqual(emb1[i][j], emb_right[i][j])
    del os.environ["TF_EMBEDDING_FBJ_OPT"]
  
  def testCPUFbjOptWithBloomFilter(self):
    print("testCPUFbjOptWithBloomFilter")
    os.environ["TF_EMBEDDING_FBJ_OPT"] = "True"
    var = variable_scope.get_embedding_variable("var_1",
            embedding_dim = 3,
            initializer=init_ops.ones_initializer(dtypes.float32),
            ev_option = variables.EmbeddingVariableOption(filter_option=variables.CBFFilter(
                                      filter_freq=3,
                                      max_element_size = 5,
                                      false_positive_probability = 0.01)))
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
        self.assertEqual(val, .0)
      emb1, top, l = sess.run([emb, train_op, loss])
      for val in emb1.tolist()[0]:
        self.assertNotEqual(val, 1.0)
    del os.environ["TF_EMBEDDING_FBJ_OPT"]

  def testSetInitializedWithoutRestore(self):
    print("testSetInitializedWithoutRestore")
    with ops.device("/cpu:0"):
      var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3)
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([1], dtypes.int64))
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    gs = training_util.get_or_create_global_step()
    opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    saver = saver_module.Saver(sharded=True)
    with self.test_session() as sess:
      result = sess.run(var._is_initialized_op)
      self.assertEqual(False, result)
      sess.run([init])
      result = sess.run(var._is_initialized_op)
      self.assertEqual(True, result)

  def testSetInitializedWithRestore(self):
    print("testSetInitializedWitRestore")
    checkpoint_directory = self.get_temp_dir()
    ckpt_path = os.path.join(checkpoint_directory, "model.ckpt")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1,2 ,3], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        sess.run([init])
        sess.run(train_op)
        saver.save(sess, ckpt_path)

    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3)
      emb = embedding_ops.embedding_lookup(var, math_ops.cast([1, 2, 3], dtypes.int64))
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
      with self.test_session(graph=g) as sess:
        result = sess.run(var._is_initialized_op)
        self.assertEqual(False, result)
        sess.run([var._initializer_for_restore])
        result = sess.run(var._is_initialized_op)
        self.assertEqual(False, result)

        saver.restore(sess, ckpt_path)
        result = sess.run(var._is_initialized_op)
        self.assertEqual(True, result)

  def testCountsTensor(self):
    os.environ["TF_RECORD_FREQ"] = "1"
    checkpoint_directory = self.get_temp_dir()
    ckpt_path = os.path.join(checkpoint_directory, "model.ckpt")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3)
      sp1 = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([0,0,0,1,1,2], dtypes.int64),
                      dense_shape=[6, 1])
      sp2 = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([3,3,3,4,4,1], dtypes.int64),
                      dense_shape=[6, 1])
      emb1 = embedding_ops.embedding_lookup_sparse(var, sp1, None)
      emb2 = embedding_ops.embedding_lookup_sparse(var, sp2, None)
      emb = emb1 + emb2
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
    with self.test_session(graph=g) as sess:
      sess.run([init])
      sess.run(train_op)
      saver.save(sess, ckpt_path)

    for name, shape in checkpoint_utils.list_variables(ckpt_path):
      if name == "var_1-freqs":
        value = checkpoint_utils.load_variable(ckpt_path, name)
        self.assertAllEqual(value, [3, 3, 1, 3, 2])
  
  def testCountsWithSparseAndDenseTensor(self):
    os.environ["TF_RECORD_FREQ"] = "1"
    checkpoint_directory = self.get_temp_dir()
    ckpt_path = os.path.join(checkpoint_directory, "model.ckpt")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3)
      sp1 = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([0,0,0,1,1,2], dtypes.int64),
                      dense_shape=[6, 1])
      ids = constant_op.constant([3,3,3,4,4,1], dtype=dtypes.int64)
      emb1 = embedding_ops.embedding_lookup_sparse(var, sp1, None)
      emb2 = embedding_ops.embedding_lookup(var, ids)
      emb = emb1 + emb2
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = adagrad_decay.AdagradDecayOptimizer(0.1, gs)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
    with self.test_session(graph=g) as sess:
      sess.run([init])
      sess.run(train_op)
      saver.save(sess, ckpt_path)

    for name, shape in checkpoint_utils.list_variables(ckpt_path):
      if name == "var_1-freqs":
        value = checkpoint_utils.load_variable(ckpt_path, name)
        self.assertAllEqual(value, [3, 3, 1, 3, 2])
  
  def testCountsTensorWithGradientDescent(self):
    os.environ["TF_RECORD_FREQ"] = "1"
    checkpoint_directory = self.get_temp_dir()
    ckpt_path = os.path.join(checkpoint_directory, "model.ckpt")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3)
      sp1 = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([0,0,0,1,1,2], dtypes.int64),
                      dense_shape=[6, 1])
      sp2 = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([3,3,3,4,4,1], dtypes.int64),
                      dense_shape=[6, 1])
      emb1 = embedding_ops.embedding_lookup_sparse(var, sp1, None)
      emb2 = embedding_ops.embedding_lookup_sparse(var, sp2, None)
      emb = emb1 + emb2
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = gradient_descent.GradientDescentOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
    with self.test_session(graph=g) as sess:
      sess.run([init])
      sess.run(train_op)
      saver.save(sess, ckpt_path)

    for name, shape in checkpoint_utils.list_variables(ckpt_path):
      if name == "var_1-freqs":
        value = checkpoint_utils.load_variable(ckpt_path, name)
        self.assertAllEqual(value, [3, 3, 1, 3, 2])

    del os.environ["TF_RECORD_FREQ"]
  
  def testCountsDenseAndSparseTensorWithGradientDescent(self):
    os.environ["TF_RECORD_FREQ"] = "1"
    checkpoint_directory = self.get_temp_dir()
    ckpt_path = os.path.join(checkpoint_directory, "model.ckpt")
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      var = variable_scope.get_embedding_variable("var_1",
          embedding_dim = 3)
      sp1 = sparse_tensor.SparseTensor(
                      indices=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],
                      values=math_ops.cast([0,0,0,1,1,2], dtypes.int64),
                      dense_shape=[6, 1])
      ids = constant_op.constant([3,3,3,4,4,1], dtype=dtypes.int64)
      emb1 = embedding_ops.embedding_lookup_sparse(var, sp1, None)
      emb2 = embedding_ops.embedding_lookup(var, ids)
      emb = emb1 + emb2
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')
      gs = training_util.get_or_create_global_step()
      opt = gradient_descent.GradientDescentOptimizer(0.1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v)
      saver = saver_module.Saver(sharded=True)
      init = variables.global_variables_initializer()
    with self.test_session(graph=g) as sess:
      sess.run([init])
      sess.run(train_op)
      saver.save(sess, ckpt_path)

    for name, shape in checkpoint_utils.list_variables(ckpt_path):
      if name == "var_1-freqs":
        value = checkpoint_utils.load_variable(ckpt_path, name)
        self.assertAllEqual(value, [3, 3, 1, 3, 2])

    del os.environ["TF_RECORD_FREQ"]

if __name__ == "__main__":
  googletest.main()
