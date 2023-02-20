"""Tests for tensorflow.ops.embedding_variable GPU version."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.platform import googletest
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import kv_variable_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import config
from tensorflow.core.framework.embedding import config_pb2
from tensorflow.python.training import training_util
from tensorflow.python.training import adagrad
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2

class GroupEmbeddingGPUTest(test_util.TensorFlowTestCase):
    
  @test_util.run_gpu_only
  def testMultiKvResourceGather(self):
    print("testMultiKvResourceGather")
    def runTestAdagrad(embedding_weights, indices, combiners):
        emb = embedding_ops.group_embedding_lookup_sparse(embedding_weights, indices, combiners)
        contcat_emb = array_ops.concat(emb, axis=-1)
        fun = math_ops.multiply(contcat_emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v)
        init = variables.global_variables_initializer()
        with self.test_session(use_gpu=True, force_gpu=True) as sess:
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
            sess.run([init])
            r, _, _ = sess.run([emb, train_op,loss])
            return r

    with ops.device('/GPU:0'):
      emb_var_0 = variable_scope.get_embedding_variable("emb_var_0",
            embedding_dim = 8,
            initializer=init_ops.ones_initializer(dtypes.float32))
      
      emb_var_1 = variable_scope.get_embedding_variable("emb_var_1",
            embedding_dim = 16,
            initializer=init_ops.ones_initializer(dtypes.float32))

    indices_0 = sparse_tensor.SparseTensor(
        indices=ops.convert_to_tensor([[0, 0], [1, 1], [2, 0], [2, 1], [3, 2]], dtype=dtypes.int64),
        values=ops.convert_to_tensor([1, 1, 3, 4, 5], dtype=dtypes.int64),
        dense_shape=[4, 3])

    indices = [indices_0 for _ in range(2)]
    ev_weights = [emb_var_0, emb_var_1]
    combiners = ["mean", "sum"]

    ev_result = runTestAdagrad(ev_weights, indices, combiners)
    for i in range(4):
        if i == 2:
            for j in range(16):
                self.assertEqual(ev_result[1].tolist()[i][j], 2)
        else:
            for j in range(16):
                self.assertEqual(ev_result[1].tolist()[i][j], 1)

    for i in range(4):
        for j in range(8):
            self.assertEqual(ev_result[0].tolist()[i][j], 1)
  
  @test_util.run_gpu_only
  def testMultiEmbeddingSparseLookUp(self):
    print("testMultiEmbeddingSparseLookUp")
    def runTestAdagrad(embedding_weights, indices, combiners):
        emb = embedding_ops.group_embedding_lookup_sparse(embedding_weights, indices, combiners)
        contcat_emb = array_ops.concat(emb, axis=-1)
        fun = math_ops.multiply(contcat_emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v)
        init = variables.global_variables_initializer()
        with self.test_session(use_gpu=True, force_gpu=True) as sess:
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
            sess.run([init])
            r, _, _ = sess.run([emb, train_op,loss])
            return r

    with ops.device('/GPU:0'):
      
      var_0 = variable_scope.get_variable("var_0",
                            initializer=init_ops.ones_initializer(dtypes.float32),
                            shape=(1000, 8))
      var_1 = variable_scope.get_variable("var_1",
                            initializer=init_ops.ones_initializer(dtypes.float32),
                           shape=(1000, 16))

    indices_0 = sparse_tensor.SparseTensor(
        indices=ops.convert_to_tensor([[0, 0], [1, 1], [2, 0], [2, 1], [3, 2]], dtype=dtypes.int64),
        values=ops.convert_to_tensor([1, 1, 3, 4, 5], dtype=dtypes.int64),
        dense_shape=[4, 3])
    

    indices = [indices_0 for _ in range(2)]
    var_weights = [var_0, var_1]
    combiners = ["mean", "sum"]

    var_result = runTestAdagrad(var_weights, indices, combiners)
    for i in range(4):
        if i == 2:
            for j in range(16):
                self.assertEqual(var_result[1].tolist()[i][j], 2)
        else:
            for j in range(16):
                self.assertEqual(var_result[1].tolist()[i][j], 1)

    for i in range(4):
        for j in range(8):
            self.assertEqual(var_result[0].tolist()[i][j], 1)
  
  @test_util.run_gpu_only
  def testMultiKvResourceGatherEqualMultiEmbeddingSparseLookUp(self):
    print("testMultiKvResourceGather")
    def runTestAdagrad(embedding_weights, indices, combiners):
        emb = embedding_ops.group_embedding_lookup_sparse(embedding_weights, indices, combiners)
        contcat_emb = array_ops.concat(emb, axis=-1)
        fun = math_ops.multiply(contcat_emb, 2.0, name='multiply')
        loss = math_ops.reduce_sum(fun, name='reduce_sum')
        gs = training_util.get_or_create_global_step()
        opt = adagrad.AdagradOptimizer(0.1)
        g_v = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(g_v)
        init = variables.global_variables_initializer()
        with self.test_session(use_gpu=True, force_gpu=True) as sess:
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
            sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
            sess.run([init])
            r, _, _ = sess.run([emb, train_op,loss])
            r, _, _ = sess.run([emb, train_op,loss])
            r, _, _ = sess.run([emb, train_op,loss])
            r, _, _ = sess.run([emb, train_op,loss])
            r, _, _ = sess.run([emb, train_op,loss])
            return r

    with ops.device('/GPU:0'):
      emb_var_1 = variable_scope.get_embedding_variable("emb_var_0",
            embedding_dim = 16,
            initializer=init_ops.ones_initializer(dtypes.float32))
      
      emb_var_2 = variable_scope.get_embedding_variable("emb_var_1",
            embedding_dim = 16,
            initializer=init_ops.ones_initializer(dtypes.float32))
      
      var_0 = variable_scope.get_variable("var_0",
                            initializer=init_ops.ones_initializer(dtypes.float32),
                            shape=(1000, 16))
      var_1 = variable_scope.get_variable("var_1",
                            initializer=init_ops.ones_initializer(dtypes.float32),
                           shape=(1000, 16))

    indices_0 = sparse_tensor.SparseTensor(
        indices=ops.convert_to_tensor([[0, 0], [1, 1], [2, 0], [2, 1], [3, 2]], dtype=dtypes.int64),
        values=ops.convert_to_tensor([1, 1, 3, 4, 5], dtype=dtypes.int64),
        dense_shape=[4, 3])

    indices = [indices_0 for _ in range(4)]
    weights = [emb_var_1, emb_var_2, var_0, var_1]
    combiners = ["mean", "sum", "mean", "sum"]


    ev_result = runTestAdagrad(weights, indices, combiners)

    for i in range(2):
        for j in range(0, 4):
            for k in range(0, 16):
                self.assertNear(ev_result[i].tolist()[j][k], ev_result[2+i].tolist()[j][k], 1e-05)
  
  @test_util.run_gpu_only
  def testMultiKvResourceGatherForSparseColumnEmbeddingCol(self):
    with feature_column_v2.group_embedding_column_scope(name="test"):
        ad_columns = feature_column_v2.categorical_column_with_embedding(key="ad_emb", dtype=dtypes.int64,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)))
        ad_weights = feature_column_v2.embedding_column(categorical_column=ad_columns,
            dimension=8,
            initializer=init_ops.ones_initializer(dtypes.float32))

        user_columns = feature_column_v2.categorical_column_with_embedding(key="user_emb", dtype=dtypes.int64,
            ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.HBM)))
        user_weights = feature_column_v2.embedding_column(categorical_column=user_columns,
            dimension=16,
            initializer=init_ops.ones_initializer(dtypes.float32))

    ids={}
    ids["ad_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,1],[2,2],[3,3],[4,3]], values=math_ops.cast([1,2,3,4,5], dtypes.int64), dense_shape=[5, 4])
    ids["user_emb"] = sparse_tensor.SparseTensor(indices=[[0,0],[1,1],[2,2],[2,3],[4,3]], values=math_ops.cast([1,2,3,4,5], dtypes.int64), dense_shape=[5, 4])

    emb = feature_column.input_layer(features=ids, feature_columns=[ad_weights, user_weights])

    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')
    opt = adagrad.AdagradOptimizer(0.1)
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

if __name__ == "__main__":
    config.enable_group_embedding(fusion_type="localized")
    googletest.main()