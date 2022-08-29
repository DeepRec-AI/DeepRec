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
"""Tests for checkpoints tools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util
from tensorflow.python.training.tracking import util as trackable_utils


def _create_checkpoints(sess, checkpoint_dir):
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  v1 = variable_scope.get_variable("var1", [1, 10])
  v2 = variable_scope.get_variable("var2", [10, 10])
  v3 = variable_scope.get_variable("var3", [100, 100])
  with variable_scope.variable_scope("useful_scope"):
    v4 = variable_scope.get_variable("var4", [9, 9])
  sess.run(variables.global_variables_initializer())
  v1_value, v2_value, v3_value, v4_value = sess.run([v1, v2, v3, v4])
  saver = saver_lib.Saver()
  saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  return v1_value, v2_value, v3_value, v4_value


def _create_ev_checkpoints(sess, checkpoint_dir, steps_to_live = None,
                             train_steps_before_save=2):
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  with variable_scope.variable_scope("useful_scope"):
    ev1 = variable_scope.get_embedding_variable("ev1", embedding_dim=8, steps_to_live=steps_to_live)
    ev2 = variable_scope.get_embedding_variable("ev2", embedding_dim=8,
      partitioner=partitioned_variables.fixed_size_partitioner(4), steps_to_live=steps_to_live)
  emb_1 = embedding_ops.embedding_lookup(ev1,
    array_ops.constant([0,1,2,3,4], dtype=dtypes.int64))
  emb_2 = embedding_ops.embedding_lookup(ev2,
    array_ops.constant([5,6,7,8,9], dtype=dtypes.int64))
  emb = array_ops.concat([emb_1, emb_2], axis=1)
  loss = math_ops.reduce_sum(emb)
  opt = adagrad.AdagradOptimizer(0.1)
  g_v = opt.compute_gradients(loss)
  gs = training_util.get_or_create_global_step()
  train_op = opt.apply_gradients(g_v, global_step=gs)
  saver = saver_lib.Saver()
  sess.run(variables.global_variables_initializer())
  for i in range(0, train_steps_before_save):
    sess.run([emb, loss, train_op])
  saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  sess.run([loss, train_op])
  l = sess.run(loss)
  return l


def _create_partition_checkpoints(sess, checkpoint_dir):
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  with variable_scope.variable_scope("scope"):
    v1 = variable_scope.get_variable(
        name="var1",
        shape=[100, 100],
        initializer=init_ops.truncated_normal_initializer(0.5),
        partitioner=partitioned_variables.min_max_variable_partitioner(
            max_partitions=5, axis=0, min_slice_size=8 << 10))
  sess.run(variables.global_variables_initializer())
  v1_value = sess.run(v1._get_variable_list())
  saver = saver_lib.Saver()
  saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  return v1_value


class CheckpointsTest(test.TestCase):

  def testNoCheckpoints(self):
    checkpoint_dir = self.get_temp_dir() + "/no_checkpoints"
    with self.assertRaises(errors_impl.OpError):
      self.assertAllEqual(
          checkpoint_utils.load_variable(checkpoint_dir, "var1"), [])

  def testNoTensor(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _, _, _, _ = _create_checkpoints(session, checkpoint_dir)
    with self.assertRaises(errors_impl.OpError):
      self.assertAllEqual(
          checkpoint_utils.load_variable(checkpoint_dir, "var5"), [])

  def testGetTensor(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var1"), v1)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var2"), v2)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var3"), v3)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "useful_scope/var4"), v4)

  def testGetAllVariables(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _create_checkpoints(session, checkpoint_dir)
    self.assertEqual(
        checkpoint_utils.list_variables(checkpoint_dir),
        [("useful_scope/var4", [9, 9]), ("var1", [1, 10]), ("var2", [10, 10]),
         ("var3", [100, 100])])

  def testInitFromCheckpoint(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          my1 = variable_scope.get_variable("my1", [1, 10])
          with variable_scope.variable_scope("some_other_scope"):
            my2 = variable_scope.get_variable("my2", [10, 10])
            with variable_scope.variable_scope("other_useful_scope"):
              my4 = variable_scope.get_variable("var4", [9, 9])
        my3 = variable_scope.get_variable("my3", [100, 100])

        checkpoint_utils.init_from_checkpoint(checkpoint_dir, {
            "var1": "some_scope/my1",
            "useful_scope/": "some_scope/some_other_scope/other_useful_scope/",
        })
        checkpoint_utils.init_from_checkpoint(checkpoint_dir, {
            "var2": "some_scope/some_other_scope/my2",
            "var3": my3,
        })

        session.run(variables.global_variables_initializer())
        self.assertAllEqual(my1.eval(session), v1)
        self.assertAllEqual(my2.eval(session), v2)
        self.assertAllEqual(my3.eval(session), v3)
        self.assertAllEqual(my4.eval(session), v4)

        # Check that tensors are not explicitly in the graph.
        self.assertLess(len(str(session.graph.as_graph_def())), 29000)

  def testInitialValueComesFromCheckpoint(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, _, _, _ = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope(
            "some_scope", initializer=init_ops.zeros_initializer()):
          my1 = variable_scope.get_variable("my1", [1, 10])

        before = my1.initialized_value()

        checkpoint_utils.init_from_checkpoint(checkpoint_dir, {"var1": my1})

        after = my1.initialized_value()

        self.assertAllEqual(session.run(before), [[0.0] * 10])
        self.assertAllEqual(session.run(after), v1)

        session.run(variables.global_variables_initializer())

        self.assertAllEqual(session.run(my1), v1)
        self.assertAllEqual(session.run(my1.initialized_value()), v1)
        self.assertAllClose(session.run(before), v1)
        self.assertAllClose(session.run(after), v1)
        with self.assertRaises(AssertionError):
          self.assertAllClose(v1, [[0.0] * 10])

  def testInitWithScopeDoesNotCaptureSuffixes(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _, _, _, v4 = _create_checkpoints(session, checkpoint_dir)

    with ops.Graph().as_default() as g:
      with variable_scope.variable_scope("useful_scope"):
        my4 = variable_scope.get_variable("var4", [9, 9])
      with variable_scope.variable_scope("useful_scope_1"):
        my5_init = [[1.0, 2.0], [3.0, 4.0]]
        my5 = variable_scope.get_variable("var5", initializer=my5_init)

      checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                            {"useful_scope/": "useful_scope/"})
      with self.session(graph=g) as session:
        session.run(variables.global_variables_initializer())
        self.assertAllEqual(my4.eval(session), v4)
        self.assertAllEqual(my5.eval(session), my5_init)

  def testRestoreRunsOnSameDevice(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _create_checkpoints(session, checkpoint_dir)

    with ops.Graph().as_default():
      with ops.device("/job:ps"):
        with variable_scope.variable_scope("useful_scope"):
          variable_scope.get_variable("var4", [9, 9])

      checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                            {"useful_scope/": "useful_scope/"})

  def testInitFromRootCheckpoint(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          my1 = variable_scope.get_variable("var1", [1, 10])
          my2 = variable_scope.get_variable("var2", [10, 10])
          my3 = variable_scope.get_variable("var3", [100, 100])
          with variable_scope.variable_scope("useful_scope"):
            my4 = variable_scope.get_variable("var4", [9, 9])

        checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                              {"/": "some_scope/",})

        session.run(variables.global_variables_initializer())
        self.assertAllEqual(my1.eval(session), v1)
        self.assertAllEqual(my2.eval(session), v2)
        self.assertAllEqual(my3.eval(session), v3)
        self.assertAllEqual(my4.eval(session), v4)

  def testInitToRootCheckpoint(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        my1 = variable_scope.get_variable("var1", [1, 10])
        my2 = variable_scope.get_variable("var2", [10, 10])
        my3 = variable_scope.get_variable("var3", [100, 100])
        with variable_scope.variable_scope("useful_scope"):
          my4 = variable_scope.get_variable("var4", [9, 9])

        checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                              {"/": "/",})

        session.run(variables.global_variables_initializer())
        self.assertAllEqual(my1.eval(session), v1)
        self.assertAllEqual(my2.eval(session), v2)
        self.assertAllEqual(my3.eval(session), v3)
        self.assertAllEqual(my4.eval(session), v4)

  def testInitFromPartitionVar(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1 = _create_partition_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          my1 = variable_scope.get_variable(
              name="my1",
              shape=[100, 100],
              initializer=init_ops.zeros_initializer(),
              partitioner=partitioned_variables.min_max_variable_partitioner(
                  max_partitions=5, axis=0, min_slice_size=8 << 10))
          my1_var_list = my1._get_variable_list()
        # Create another variable with different partitions than the variable in
        # the checkpoint.
        with variable_scope.variable_scope("some_other_scope"):
          my2 = variable_scope.get_variable(
              name="var1",
              shape=[100, 100],
              initializer=init_ops.zeros_initializer(),
              partitioner=partitioned_variables.min_max_variable_partitioner(
                  max_partitions=5, axis=0, min_slice_size=16 << 10))
          my2_var_list = my2._get_variable_list()

        checkpoint_utils.init_from_checkpoint(checkpoint_dir, {
            "scope/var1": "some_scope/my1",
            "scope/": "some_other_scope/"})

        session.run(variables.global_variables_initializer())
        my1_values = session.run(my1_var_list)
        self.assertAllEqual(my1_values, v1)
        my2_values = session.run(my2_var_list)
        # Verify we created different number of partitions.
        self.assertNotEquals(len(my2_values), len(v1))
        # Verify the values were correctly initialized inspite of different
        # partitions.
        full_my2_values = np.concatenate(my2_values, axis=0)
        full_v1_values = np.concatenate(v1, axis=0)
        self.assertAllEqual(full_my2_values, full_v1_values)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          my1 = variable_scope.get_variable(
              name="my1",
              shape=[100, 100],
              initializer=init_ops.truncated_normal_initializer(0.5),
              partitioner=partitioned_variables.min_max_variable_partitioner(
                  max_partitions=5, axis=0, min_slice_size=8 << 10))
          my1_var_list = my1._get_variable_list()

        checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                              {"scope/var1": my1_var_list,})

        session.run(variables.global_variables_initializer())
        my1_values = session.run(my1_var_list)
        self.assertAllEqual(my1_values, v1)

  def testInitFromCheckpointMissing(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _, _, _, _ = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          _ = variable_scope.get_variable("my1", [10, 10])
          _ = variable_scope.get_variable(
              "my2", [1, 10],
              dtype=dtypes.int64,
              initializer=init_ops.zeros_initializer())

        # No directory.
        with self.assertRaises(errors_impl.OpError):
          checkpoint_utils.init_from_checkpoint("no_dir",
                                                {"var1": "some_scope/my1"})

        # No variable in checkpoint.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                                {"no_var": "some_scope/my1"})

        # No variable in the graph.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                                {"var3": "some_scope/no_var"})

        # Shape mismatch.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                                {"var1": "some_scope/my1"})

        # Variable 'my1' and 'my2' are missing in given checkpoint scope.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(
              checkpoint_dir, {"useful_scope/": "some_scope/"})

        # Mapping is not to scope name.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                                {"useful_scope": "some_scope/"})

  def testNoAdditionalReadOpsForResourceVariables(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, _, _, _ = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        my1 = resource_variable_ops.ResourceVariable([[0.0] * 10], name="my1")

        with ops.name_scope("init_from_checkpoint"):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir, {"var1": my1})

        # Basic sanity checks:
        session.run(variables.global_variables_initializer())
        self.assertAllEqual(session.run(my1), v1)

    ops_in_init_from_checkpoint_scope = [
        op for op in g.get_operations()
        if (op.name.startswith("init_from_checkpoint/") and
            not op.name.startswith("init_from_checkpoint/checkpoint_initializer"
                                  ) and
            op.type != "AssignVariableOp" and
            op.type != "Identity")
    ]
    self.assertEqual(ops_in_init_from_checkpoint_scope, [])

  def testInitWithScopeEmbeddingVariable(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      loss = _create_ev_checkpoints(session, checkpoint_dir)

    with ops.Graph().as_default() as g:
      with variable_scope.variable_scope("useful_scope"):
        ev1 = variable_scope.get_embedding_variable("ev1", embedding_dim=8)
      with variable_scope.variable_scope("useful_scope"):
        ev2 = variable_scope.get_embedding_variable("ev2", embedding_dim=8,
          partitioner=partitioned_variables.fixed_size_partitioner(4))

      emb_1 = embedding_ops.embedding_lookup(ev1,
        array_ops.constant([0,1,2,3,4], dtype=dtypes.int64))
      emb_2 = embedding_ops.embedding_lookup(ev2,
        array_ops.constant([5,6,7,8,9], dtype=dtypes.int64))
      emb = array_ops.concat([emb_1, emb_2], axis=1)
      my_loss = math_ops.reduce_sum(emb)
      opt = adagrad.AdagradOptimizer(0.1)
      g_v = opt.compute_gradients(my_loss)
      train_op = opt.apply_gradients(g_v)
      checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                            {"useful_scope/": "useful_scope/"})
      with self.session(graph=g) as session:
        session.run(variables.global_variables_initializer())
        session.run([emb, my_loss, train_op])
        self.assertAllEqual(session.run(my_loss), loss)

  def testResetVersionWithScopeEmbeddingVariable(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      loss = _create_ev_checkpoints(session, checkpoint_dir, steps_to_live=20)

    with ops.Graph().as_default() as g:
      with variable_scope.variable_scope("useful_scope"):
        ev1 = variable_scope.get_embedding_variable("ev1", embedding_dim=8, steps_to_live=20)
      with variable_scope.variable_scope("useful_scope"):
        ev2 = variable_scope.get_embedding_variable("ev2", embedding_dim=8,
          partitioner=partitioned_variables.fixed_size_partitioner(4), steps_to_live=20)
      saver = saver_lib.Saver()
      checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                            {"useful_scope/": "useful_scope/"}, reset_version=True)
      with self.session(graph=g) as session:
        session.run(variables.global_variables_initializer())
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        checkpoint_state_name = "checkpoint"
        saver.save(
          session,
          checkpoint_prefix,
          global_step=0,
          latest_filename=checkpoint_state_name)
        for name, shape in checkpoint_utils.list_variables(checkpoint_dir):
          if "-versions" in name:
            version_list = checkpoint_utils.load_variable(checkpoint_dir, name)
            for val in version_list:
              self.assertEqual(val, 0)

  def testWithoutResetVersionWithScopeEmbeddingVariable(self):
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      loss = _create_ev_checkpoints(session, checkpoint_dir, steps_to_live=20)

    with ops.Graph().as_default() as g:
      with variable_scope.variable_scope("useful_scope"):
        ev1 = variable_scope.get_embedding_variable("ev1", embedding_dim=8, steps_to_live=20)
      with variable_scope.variable_scope("useful_scope"):
        ev2 = variable_scope.get_embedding_variable("ev2", embedding_dim=8,
          partitioner=partitioned_variables.fixed_size_partitioner(4), steps_to_live=20)
      saver = saver_lib.Saver()
      checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                            {"useful_scope/": "useful_scope/"})
      with self.session(graph=g) as session:
        session.run(variables.global_variables_initializer())
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        checkpoint_state_name = "checkpoint"
        saver.save(
          session,
          checkpoint_prefix,
          global_step=0,
          latest_filename=checkpoint_state_name)
        for name, shape in checkpoint_utils.list_variables(checkpoint_dir):
          if "-versions" in name:
            version_list = checkpoint_utils.load_variable(checkpoint_dir, name)
            for val in version_list:
              self.assertEqual(val, 1)




class CheckpointIteratorTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testReturnsEmptyIfNoCheckpointsFound(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), "no_checkpoints_found")

    num_found = 0
    for _ in checkpoint_utils.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 0)

  @test_util.run_in_graph_and_eager_modes
  def testReturnsSingleCheckpointIfOneCheckpointFound(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), "one_checkpoint_found")
    if not gfile.Exists(checkpoint_dir):
      gfile.MakeDirs(checkpoint_dir)

    save_path = os.path.join(checkpoint_dir, "model.ckpt")

    a = resource_variable_ops.ResourceVariable(5)
    self.evaluate(a.initializer)
    checkpoint = trackable_utils.Checkpoint(a=a)
    checkpoint.save(file_prefix=save_path)

    num_found = 0
    for _ in checkpoint_utils.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 1)

  @test_util.run_v1_only("Tests v1-style checkpoint sharding")
  def testReturnsSingleCheckpointIfOneShardedCheckpoint(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  "one_checkpoint_found_sharded")
    if not gfile.Exists(checkpoint_dir):
      gfile.MakeDirs(checkpoint_dir)

    global_step = variables.Variable(0, name="v0")

    # This will result in 3 different checkpoint shard files.
    with ops.device("/cpu:0"):
      variables.Variable(10, name="v1")
    with ops.device("/cpu:1"):
      variables.Variable(20, name="v2")

    saver = saver_lib.Saver(sharded=True)

    with session_lib.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as session:

      session.run(variables.global_variables_initializer())
      save_path = os.path.join(checkpoint_dir, "model.ckpt")
      saver.save(session, save_path, global_step=global_step)

    num_found = 0
    for _ in checkpoint_utils.checkpoints_iterator(checkpoint_dir, timeout=0):
      num_found += 1
    self.assertEqual(num_found, 1)

  @test_util.run_in_graph_and_eager_modes
  def testTimeoutFn(self):
    timeout_fn_calls = [0]
    def timeout_fn():
      timeout_fn_calls[0] += 1
      return timeout_fn_calls[0] > 3

    results = list(
        checkpoint_utils.checkpoints_iterator(
            "/non-existent-dir", timeout=0.1, timeout_fn=timeout_fn))
    self.assertEqual([], results)
    self.assertEqual(4, timeout_fn_calls[0])


@test_util.run_all_in_graph_and_eager_modes
class WaitForNewCheckpointTest(test.TestCase):

  def testReturnsNoneAfterTimeout(self):
    start = time.time()
    ret = checkpoint_utils.wait_for_new_checkpoint(
        "/non-existent-dir", "foo", timeout=1.0, seconds_to_sleep=0.5)
    end = time.time()
    self.assertIsNone(ret)

    # We've waited one second.
    self.assertGreater(end, start + 0.5)

    # The timeout kicked in.
    self.assertLess(end, start + 1.1)


if __name__ == "__main__":
  test.main()
