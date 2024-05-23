# Copyright (c) 2017, Alibaba Inc.
# All right reserved.
#
# Author: Chen Ding <cnady.dc@alibaba-inc.com>
# Created: 2018/03/26
# Description:
# ==============================================================================

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
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import gen_kv_variable_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.training import ftrl
from tensorflow.python.training import adam
from tensorflow.python.training import adagrad
from tensorflow.python.training import training_util
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import incremental_saver as incr_saver_module 
from tensorflow.python.training import training_util
from tensorflow.python.ops import variables
from tensorflow.contrib.layers.python.layers import embedding_ops as emb_ops
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.framework.python.framework import checkpoint_utils

class IncrSaveRestoreTest(test_util.TensorFlowTestCase):
  def testBasicIncr(self):
    print("testBasicIncr")
    ckpt_path = os.path.join(self.get_temp_dir(), "model.ckpt")
    with ops.device("/device:CPU:0"):
      apply_incr = gen_io_ops.record_sparse_indices(math_ops.cast([0,1,2,5,6,7], dtypes.int64), "v0")
    v0 = variables.Variable(10.0, name="v0")
    saver = saver_module.Saver({"v0":v0})
    activate_op = gen_io_ops.activate_sparse_recorder(["v0"])

    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(activate_op)
      sess.run(apply_incr)
      saver.save(sess, ckpt_path, global_step=12345)
      sess.run(apply_incr)
 
  
  def testSparseEvIncrSaveRestore(self):
    tmppath = self.get_temp_dir()
    #tmppath = "/tmp/fcev"
    #os.system("rm -rf " + tmppath)
    full_ckpt_path = os.path.join(tmppath, "model.ckpt")
    incr_ckpt_path = os.path.join(tmppath, "incr.ckpt")
    var = variable_scope.get_embedding_variable("var_ev1", embedding_dim=3, key_dtype=dtypes.int64,
            initializer=init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    with ops.device("/device:CPU:0"):
      apply_incr = gen_io_ops.record_sparse_indices(math_ops.cast([0,1,2,5,6,7], dtypes.int64), "var_ev1")
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    ev_var_name = "var_ev1"
    incr_save_op = gen_io_ops.incr_save(incr_ckpt_path, [ev_var_name], [], [True],[var.handle])
    activate_op = gen_io_ops. activate_sparse_recorder(["var_ev1"])

    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(emb)
      sess.run(apply_incr)
      sess.run(activate_op)
      saver.save(sess, full_ckpt_path, global_step=10)
      sess.run(apply_incr)
      sess.run(incr_save_op)
      incr_restore_op = gen_kv_variable_ops.kv_resource_incr_import(incr_ckpt_path, var.handle, var._shared_name, 
        ops.convert_to_tensor(var.invalid_key, preferred_dtype=dtypes.int64), variables._try_guard_against_uninitialized_dependencies(var.name, var.initial_value))
      print(sess.run(incr_restore_op))
   

  def testSparseNormIncrSaveRestoreInt64(self):
    tmppath = self.get_temp_dir()
    print(tmppath)
    full_ckpt_path = os.path.join(tmppath, "model.ckpt")
    incr_ckpt_path = os.path.join(tmppath, "incr.ckpt")
    var = variable_scope.get_variable("var_norm1", shape=[20,3],  initializer=init_ops.constant_initializer(1))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    with ops.device("/device:CPU:0"):
      apply_incr = gen_io_ops.record_sparse_indices(math_ops.cast([0,1,2,5,6,7], dtypes.int64), "var_norm1")
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    incr_save_op = gen_io_ops.incr_save(incr_ckpt_path, ["var_norm1"], [], [True], [var])
    activate_op = gen_io_ops. activate_sparse_recorder(["var_norm1"])
    with self.test_session() as sess:
      sess.run([init])
      sess.run(var)
      sess.run(emb)
      sess.run(apply_incr)
      saver.save(sess, full_ckpt_path, global_step=10)
      sess.run(activate_op)
      sess.run(apply_incr)
      sess.run(incr_save_op)
      incr_restore_op = gen_io_ops.incr_restore(incr_ckpt_path, ["var_norm1"],[], [True],
          [gen_io_ops.restore_v2(full_ckpt_path + "-10", ["var_norm1"],[""], [dtypes.float32])[0]])
      print(sess.run(incr_restore_op))
      saver.save(sess, full_ckpt_path, global_step=20)
      sess.run(incr_save_op)
      incr_restore_op = gen_io_ops.incr_restore(incr_ckpt_path, ["var_norm1"],[], [True],
          [gen_io_ops.restore_v2(full_ckpt_path + "-20", ["var_norm1"],[""], [dtypes.float32])[0]])
      print(sess.run(incr_restore_op))

  def testSparseNormIncrSaveRestoreInt32(self):
    tmppath = self.get_temp_dir()
    print(tmppath)
    full_ckpt_path = os.path.join(tmppath, "model.ckpt")
    incr_ckpt_path = os.path.join(tmppath, "incr.ckpt")
    var = variable_scope.get_variable("var_norm1", shape=[20,3],  initializer=init_ops.constant_initializer(1))
    emb = embedding_ops.embedding_lookup(var, math_ops.cast([0,1,2,5,6,7], dtypes.int32))
    with ops.device("/device:CPU:0"):
      apply_incr = gen_io_ops.record_sparse_indices(math_ops.cast([0,1,2,5,6,7], dtypes.int32), "var_norm1")
    saver = saver_module.Saver()
    init = variables.global_variables_initializer()
    incr_save_op = gen_io_ops.incr_save(incr_ckpt_path, ["var_norm1"], [], [True], [var])
    activate_op = gen_io_ops. activate_sparse_recorder(["var_norm1"])
    with self.test_session() as sess:
      sess.run([init])
      sess.run(var)
      sess.run(emb)
      sess.run(apply_incr)
      saver.save(sess, full_ckpt_path, global_step=10)
      sess.run(activate_op)
      sess.run(apply_incr)
      sess.run(incr_save_op)
      incr_restore_op = gen_io_ops.incr_restore(incr_ckpt_path, ["var_norm1"],[], [True],
          [gen_io_ops.restore_v2(full_ckpt_path + "-10", ["var_norm1"],[""], [dtypes.float32])[0]])
      print(sess.run(incr_restore_op))
      saver.save(sess, full_ckpt_path, global_step=20)
      sess.run(incr_save_op)
      incr_restore_op = gen_io_ops.incr_restore(incr_ckpt_path, ["var_norm1"],[], [True],
          [gen_io_ops.restore_v2(full_ckpt_path + "-20", ["var_norm1"],[""], [dtypes.float32])[0]])
      print(sess.run(incr_restore_op))

  
  def testMixIncrSaveRestore(self):
    tmppath = self.get_temp_dir()
    print(tmppath)
    full_ckpt_path = os.path.join(tmppath, "model.ckpt")
    incr_ckpt_path = os.path.join(tmppath, "incr.ckpt")

    var_norm = variable_scope.get_variable("var_norm1", shape=[20,3],  initializer=init_ops.constant_initializer(1))
    emb_var_norm = embedding_ops.embedding_lookup(var_norm, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    with ops.device("/device:CPU:0"):
      record_norm_incr = gen_io_ops.record_sparse_indices(math_ops.cast([0,1,2,5,6,7], dtypes.int64), "var_norm1")

    var_ev = variable_scope.get_embedding_variable("var_ev1", embedding_dim=3, key_dtype=dtypes.int64,
            initializer=init_ops.ones_initializer(dtypes.float32))
    emb_var_ev = embedding_ops.embedding_lookup(var_ev, math_ops.cast([0,1,2,5,6,7], dtypes.int64))
    with ops.device("/device:CPU:0"):
      record_ev_incr = gen_io_ops.record_sparse_indices(math_ops.cast([0,1,2,5,6,7], dtypes.int64), "var_ev1")
    activate_op = gen_io_ops. activate_sparse_recorder(["var_ev1","var_norm1"])
 
  
    saver = saver_module.Saver(sharded=True)
    init = variables.global_variables_initializer()
    incr_save_op = gen_io_ops.incr_save(incr_ckpt_path, ["var_norm1", "var_ev1"], [], [True, True], [var_norm, var_ev.handle])
    
    ev_tensors = []  
    ev_var_name = "var_ev1"
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(emb_var_norm)
      sess.run(emb_var_ev)
      sess.run(record_norm_incr)
      sess.run(record_ev_incr)
      saver.save(sess, full_ckpt_path, global_step=10)
      sess.run(activate_op)
      sess.run(record_norm_incr)
      sess.run(record_ev_incr)
      sess.run(incr_save_op)
      incr_restore_op = gen_kv_variable_ops.kv_resource_incr_import(incr_ckpt_path, var_ev.handle, var_ev._shared_name, 
        ops.convert_to_tensor(var_ev.invalid_key, preferred_dtype=dtypes.int64), variables._try_guard_against_uninitialized_dependencies(var_ev.name, var_ev.initial_value))
      print(sess.run(incr_restore_op))
      incr_restore_op = gen_io_ops.incr_restore(incr_ckpt_path, ["var_norm1"],[], [True],
          [gen_io_ops.restore_v2(full_ckpt_path + "-10", ["var_norm1"],[""], [dtypes.float32])[0]])
      print(sess.run(incr_restore_op))
  

  def testIncrementalSaveFlushNone(self):
    ev_var = variable_scope.get_embedding_variable("ev_a", embedding_dim=4, initializer=init_ops.ones_initializer(dtypes.float32))
    sparse_var = variable_scope.get_variable("sparse_var", shape=[30,4], initializer=init_ops.ones_initializer(dtypes.float32))
    dense_var = variable_scope.get_variable("dense_var", shape=[30,4], initializer=init_ops.ones_initializer(dtypes.float32))

    ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
    emb = embedding_ops.embedding_lookup(sparse_var, ids)
    emb2 = embedding_ops.embedding_lookup(ev_var, ids)
    emb = (emb + emb2)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')

    gs = training_util.get_or_create_global_step()

    opt=adagrad.AdagradOptimizer(0.1, initial_accumulator_value=1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)

    path = os.path.join(self.get_temp_dir(), "model.ckpt")
    incr_path = os.path.join(self.get_temp_dir(), ".incr/model.ckpt")
    #path = os.path.join('/tmp/1/2/3', "model.ckpt")
    saver=saver_module.Saver(sharded=True, incremental_save_restore=True)
    incr_saver=incr_saver_module.IncrementalSaver(sharded=True, saver_def=saver.saver_def, defer_build=True)
    incr_saver.build(saver._builder.filename_tensor)

    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      #print(sess.graph.as_graph_def())
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(train_op, feed_dict={"ids:0": 1})
      incr_saver.incremental_save(sess, incr_path, global_step=12345)
      for name, shape in checkpoint_utils.list_variables(incr_path + "-12345"):
        print('loading... ', name, shape, checkpoint_utils.load_variable(incr_path + "-12345", name))

  def testIncrementalSaveNormalFlush(self):
    ev_var = variable_scope.get_embedding_variable("ev_a", embedding_dim=4, initializer=init_ops.ones_initializer(dtypes.float32))
    sparse_var = variable_scope.get_variable("sparse_var", shape=[30,4], initializer=init_ops.ones_initializer(dtypes.float32))
    dense_var = variable_scope.get_variable("dense_var", shape=[30,4], initializer=init_ops.ones_initializer(dtypes.float32))

    ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
    emb = embedding_ops.embedding_lookup(sparse_var, ids)
    emb2 = embedding_ops.embedding_lookup(ev_var, ids)
    emb = (emb + emb2)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')

    gs = training_util.get_or_create_global_step()

    opt=adagrad.AdagradOptimizer(0.1, initial_accumulator_value=1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)

    path = os.path.join(self.get_temp_dir(), "model.ckpt")
    incr_path = os.path.join(self.get_temp_dir(), ".incr/model.ckpt")
    saver=saver_module.Saver(sharded=True, incremental_save_restore=True)
    incr_saver=incr_saver_module.IncrementalSaver(sharded=True, saver_def=saver.saver_def, defer_build=True)
    incr_saver.build(saver._builder.filename_tensor)

    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      #print(sess.graph.as_graph_def())
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(train_op, feed_dict={"ids:0": 1})
      saver.save(sess, path, global_step=12345)
      sess.run(train_op, feed_dict={"ids:0": 2})
      sess.run(train_op, feed_dict={"ids:0": 3})
      sess.run(train_op, feed_dict={"ids:0": 4})
      incr_saver.incremental_save(sess, incr_path, global_step=12345)
      for name, shape in checkpoint_utils.list_variables(incr_path + "-12345"):
        print('loading... ', name, shape, checkpoint_utils.load_variable(incr_path + "-12345", name))

  def testIncrementalSaveNormalFlushPartitionedVariable(self):
    
    ev_var = variable_scope.get_embedding_variable("ev_a", embedding_dim=4,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    
    sparse_var = variable_scope.get_variable("sparse_var", shape=[30,4],
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    dense_var = variable_scope.get_variable("dense_var", shape=[30,4],
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))

    ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
    emb = embedding_ops.embedding_lookup(sparse_var, ids)
    emb2 = embedding_ops.embedding_lookup(ev_var, ids)
    emb = (emb + emb2)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')

    gs = training_util.get_or_create_global_step()

    opt=adagrad.AdagradOptimizer(0.1, initial_accumulator_value=1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)

    path = os.path.join(self.get_temp_dir(), "model.ckpt")
    incr_path = os.path.join(self.get_temp_dir(), ".incr/model.ckpt")
    saver=saver_module.Saver(sharded=True, incremental_save_restore=True)
    incr_saver=incr_saver_module.IncrementalSaver(sharded=True, saver_def=saver.saver_def, defer_build=True)
    incr_saver.build(saver._builder.filename_tensor)



    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(train_op, feed_dict={"ids:0": 1})
      saver.save(sess, path, global_step=12345)
      sess.run(train_op, feed_dict={"ids:0": 2})
      sess.run(train_op, feed_dict={"ids:0": 3})
      sess.run(train_op, feed_dict={"ids:0": 4})
      incr_saver.incremental_save(sess, incr_path, global_step=12345)
      for name, shape in checkpoint_utils.list_variables(incr_path + "-12345"):
        print('loading... ', name, shape, checkpoint_utils.load_variable(incr_path + "-12345", name))

  def testIncrementalRestoreNormalFlushPartitionedVariable(self):
    ev_var = variable_scope.get_embedding_variable("ev_a", embedding_dim=4,
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    sparse_var = variable_scope.get_variable("sparse_var", shape=[30,4],
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    dense_var = variable_scope.get_variable("dense_var", shape=[30,4],
            initializer=init_ops.ones_initializer(dtypes.float32),
            partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))

    ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
    emb = embedding_ops.embedding_lookup(sparse_var, ids)
    emb2 = embedding_ops.embedding_lookup(ev_var, ids)
    emb = (emb + emb2)
    fun = math_ops.multiply(emb, 2.0, name='multiply')
    loss = math_ops.reduce_sum(fun, name='reduce_sum')

    gs = training_util.get_or_create_global_step()

    opt=adagrad.AdagradOptimizer(0.1, initial_accumulator_value=1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v, global_step=gs)

    path = os.path.join(self.get_temp_dir(), "model.ckpt")
    incr_path = os.path.join(self.get_temp_dir(), ".incr/model.ckpt")
    saver=saver_module.Saver(sharded=True, incremental_save_restore=True)
    incr_saver=incr_saver_module.IncrementalSaver(sharded=True, saver_def=saver.saver_def, defer_build=True)
    incr_saver.build(saver._builder.filename_tensor)

    init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(train_op, feed_dict={"ids:0": 1})
      saver.save(sess, path, global_step=12345)
      sess.run(train_op, feed_dict={"ids:0": 2})
      sess.run(train_op, feed_dict={"ids:0": 3})
      sess.run(train_op, feed_dict={"ids:0": 4})
      incr_saver.incremental_save(sess, incr_path, global_step=12345)
      print("PAHT", path, incr_path)
      for name, shape in checkpoint_utils.list_variables(path + "-12345"):
        print('ckpt loading... ', name, shape, checkpoint_utils.load_variable(path + "-12345", name))
      for name, shape in checkpoint_utils.list_variables(incr_path + "-12345"):
        print('loading... ', name, shape, checkpoint_utils.load_variable(incr_path + "-12345", name))
    with self.test_session() as sess:
      saver.restore(sess, path + "-12345")
      print("====incr====")
      incr_saver.incremental_restore(sess, path + "-12345", incr_path + "-12345")
      for i in [1,2,3,4]:
        for j in xrange(4):
          self.assertAlmostEqual(1.8211145, sess.run(emb, feed_dict={"ids:0": i})[j], delta=1e-05)

  def testIncrementalRestoreNormalFlushPartitionedVariableWithNormalVariable(self):
    with ops.device("/device:CPU:0"):
      ev_var = variable_scope.get_embedding_variable("ev_a", embedding_dim=4,
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      sparse_var = variable_scope.get_variable("sparse_var", shape=[30,4],
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
      dense_var = variable_scope.get_variable("dense_var", shape=[30,4],
              initializer=init_ops.ones_initializer(dtypes.float32),
              partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))

      ids = array_ops.placeholder(dtype=dtypes.int64, name='ids')
      emb = embedding_ops.embedding_lookup(sparse_var, ids)
      emb2 = embedding_ops.embedding_lookup(ev_var, ids)
      emb = (emb + emb2)
      fun = math_ops.multiply(emb, 2.0, name='multiply')
      loss = math_ops.reduce_sum(fun, name='reduce_sum')

      gs = training_util.get_or_create_global_step()

      opt=adagrad.AdagradOptimizer(0.1, initial_accumulator_value=1)
      g_v = opt.compute_gradients(loss)
      train_op = opt.apply_gradients(g_v, global_step=gs)

      path = os.path.join(self.get_temp_dir(), "model.ckpt")
      incr_path = os.path.join(self.get_temp_dir(), ".incr/model.ckpt")
      saver=saver_module.Saver(sharded=True, incremental_save_restore=True, incremental_include_normal_var=True)
      incr_saver=incr_saver_module.IncrementalSaver(sharded=True, saver_def=saver.saver_def, defer_build=True, incremental_include_normal_var=True)
      incr_saver.build(saver._builder.filename_tensor)

      init = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run(train_op, feed_dict={"ids:0": 1})
      saver.save(sess, path, global_step=12345)
      sess.run(train_op, feed_dict={"ids:0": 2})
      sess.run(train_op, feed_dict={"ids:0": 3})
      sess.run(train_op, feed_dict={"ids:0": 4})
      incr_saver.incremental_save(sess, incr_path, global_step=12345)
      print("PAHT", path, incr_path)
      for name, shape in checkpoint_utils.list_variables(path + "-12345"):
        print('ckpt loading... ', name, shape, checkpoint_utils.load_variable(path + "-12345", name))
      for name, shape in checkpoint_utils.list_variables(incr_path + "-12345"):
        print('loading... ', name, shape, checkpoint_utils.load_variable(incr_path + "-12345", name))
    with self.test_session() as sess:
      saver.restore(sess, path + "-12345")
      print("====incr====")
      incr_saver.incremental_restore(sess, path + "-12345", incr_path + "-12345")
      for i in [1,2,3,4]:
        for j in xrange(4):
          self.assertAlmostEqual(1.8211145, sess.run(emb, feed_dict={"ids:0": i})[j], delta=1e-05)

  def testIncrementalSaverForResourceVariable(self):
    variable_scope.get_variable('partitioned_res_var', shape=[100], use_resource=True, partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    variable_scope.get_variable('partitioned_var', shape=[100], use_resource=False, partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    variable_scope.get_embedding_variable('partitioned_ev', embedding_dim=100, partitioner=partitioned_variables.fixed_size_partitioner(num_shards=4))
    variable_scope.get_variable('res_var', shape=[100], use_resource=True)
    variable_scope.get_variable('var', shape=[100], use_resource=False)
    variable_scope.get_embedding_variable('ev', embedding_dim=100)
    saver = saver_module.Saver(
        sharded=True,
        save_relative_paths=True,
        incremental_save_restore=True,
    )
    saver.build()
    incr_saver = incr_saver_module._get_incremental_saver(True, saver)

  def testIncrementalSaverSaveAndRestore(self):
    tmp_path = self.get_temp_dir()
    full_ckpt_dir = os.path.join(tmp_path, "model.ckpt")
    incr_ckpt_dir = os.path.join(tmp_path, "incr.ckpt")
    full_ckpt_path = None
    incr_ckpt_path = None

    # construct graph
    emb_var = variable_scope.get_embedding_variable("emb", embedding_dim=3,
                initializer = init_ops.ones_initializer(dtypes.float32))
    emb = embedding_ops.embedding_lookup(emb_var,
            math_ops.cast([0, 1, 2, 3, 4], dtypes.int64))
    loss = math_ops.reduce_sum(emb, name = 'reduce_sum')
    opt = adagrad.AdagradOptimizer(0.1)
    g_v = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(g_v)
    init = variables.global_variables_initializer()
    saver = saver_module.Saver(sharded=True, incremental_save_restore=True)
    incr_saver = \
      incr_saver_module.IncrementalSaver(sharded=True,
                          saver_def=saver.saver_def, defer_build=True)
    incr_saver.build(saver._builder.filename_tensor)

    # generate full ckpt and incr ckpt.
    full_ckpt_value=None
    incr_ckpt_value=None
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      sess.run([train_op])
      full_ckpt_path = saver.save(sess, full_ckpt_dir, global_step = 10)
      full_ckpt_value = sess.run([emb])
      print("full_ckpt: {}".format(full_ckpt_value))
      sess.run([train_op])
      incr_ckpt_path = \
        incr_saver.incremental_save(sess, incr_ckpt_dir, global_step=20)
      incr_ckpt_value = sess.run([emb])
      print("incr_ckpt: {}".format(incr_ckpt_value))

    # check the value after restoring parameter.
    with self.test_session() as sess:
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_VAR_OPS))
      sess.run(ops.get_collection(ops.GraphKeys.EV_INIT_SLOT_OPS))
      sess.run([init])
      saver.restore(sess, full_ckpt_path)
      restore_full_ckpt_value = sess.run([emb])
      print("restore_full_ckpt: {}".format(restore_full_ckpt_value))
      incr_saver.incremental_restore(sess, full_ckpt_path, incr_ckpt_path)
      restore_incr_ckpt_value = sess.run([emb])
      print("restore_incr_ckpt: {}".format(restore_incr_ckpt_value))
      self.assertAllClose(full_ckpt_value, restore_full_ckpt_value)
      self.assertAllClose(incr_ckpt_value, restore_incr_ckpt_value)

if __name__ == "__main__":
  googletest.main()
