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
"""Tests for work queue."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Manager
import os
import random
from six import string_types
import portpicker

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import test as test_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.training import monitored_session
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.training import training_util

from tensorflow.python.ops.work_queue import WorkQueue
from tensorflow.python.ops.work_queue import LocalWorkMgr
from tensorflow.python.training import saver
from tensorflow.python.training.saver import PartialRestoreSaverBuilder

# pylint: disable=missing-docstring
class WorkQueueTest(test_lib.TestCase):
  def __init__(self, method_name='runTest'):
    super(WorkQueueTest, self).__init__(method_name)
    self._config = config_pb2.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True,
        gpu_options=config_pb2.GPUOptions(
            allow_growth=True,
            force_gpu_compatible=True))

  def test_simple(self):
    with self.test_session():
      works = [b"to", b"be", b"or", b"not", b"to", b"be"]
      num_epochs = 3
      work_queue = WorkQueue(works, num_epochs=num_epochs, shuffle=False)

      local_queue = work_queue.input_producer()
      dequeue = local_queue.dequeue()
      dequeue_many = local_queue.dequeue_many(len(works) * num_epochs)

      resources.initialize_resources(resources.shared_resources()).run()
      variables.global_variables_initializer().run()
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      local_works = dequeue_many.eval().tolist()
      self.assertEqual(
          works * num_epochs,
          [item for work in local_works for item in work])

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def test_shuffle(self):
    with self.test_session():
      works = [b"to", b"be", b"or", b"not", b"to", b"be"]
      num_epochs = 1
      work_queue = WorkQueue(works, num_epochs=num_epochs, shuffle=True)

      local_queue = work_queue.input_producer()
      dequeue = local_queue.dequeue()
      dequeue_many = local_queue.dequeue_many(len(works) * num_epochs)

      resources.initialize_resources(resources.shared_resources()).run()
      variables.global_variables_initializer().run()
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      local_works = dequeue_many.eval().tolist()
      self.assertEqual(
          sorted(works * num_epochs),
          sorted([item for work in local_works for item in work]))

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def test_slices(self):
    with self.test_session():
      works = [
          b"hdfs://path/to/my/file",
          b"hdfs://prj/tables/tbl1",
          b"hdfs://prj/tables/tbl1/pt=12",
          b"hdfs://prj/tables/tbl1/pt=12/pt1=abc"]
      num_epochs = 1
      work_queue = WorkQueue(
          works, num_epochs=num_epochs, shuffle=False, num_slices=5)

      local_queue = work_queue.input_producer()
      dequeue = local_queue.dequeue()
      dequeue_many = local_queue.dequeue_many(len(works) * num_epochs)

      resources.initialize_resources(resources.shared_resources()).run()
      variables.global_variables_initializer().run()
      variables.local_variables_initializer().run()
      threads = queue_runner_impl.start_queue_runners()

      local_works = dequeue_many.eval().tolist()
      self.assertEqual(
          works * num_epochs,
          [item for work in local_works for item in work])

      # Reached the limit.
      with self.assertRaises(errors_impl.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def test_monitored_session(self):
    ps_hosts = ["localhost:{}".format(portpicker.pick_unused_port())]
    worker_hosts = ["localhost:{}".format(portpicker.pick_unused_port())]

    def ps_fn(ps_hosts, worker_hosts):
      cluster = server_lib.ClusterSpec(
          {"ps": ps_hosts, "worker": worker_hosts})
      server = server_lib.Server(
          cluster, job_name="ps", task_index=0, start=True,
          config=self._config)
      server.join()

    def worker_fn(ps_hosts, worker_hosts, worker_id):
      cluster = server_lib.ClusterSpec(
          {"ps": ps_hosts, "worker": worker_hosts})
      worker = server_lib.Server(
          cluster, job_name="worker", task_index=worker_id, start=True,
          config=self._config)
      with ops.Graph().as_default():
        with ops.device(device_setter.replica_device_setter(cluster=cluster)):
          with ops.device('/cpu:0'):
            training_util.get_or_create_global_step()
            work_queue = WorkQueue(
                [b"fast", b"fox", b"jumps", b"over", b"lazy", b"dog"],
                num_epochs=3, shuffle=False, name='global_queue')
            local_queue = work_queue.input_producer()
            read_op = local_queue.dequeue()
        with monitored_session.MonitoredTrainingSession(
            master=worker.target) as sess:
          sess.run(read_op)
    ps_proc = Process(target=ps_fn, args=(ps_hosts, worker_hosts))
    ps_proc.start()
    worker_proc = Process(
        target=worker_fn, args=(ps_hosts, worker_hosts, 0))
    worker_proc.start()
    worker_proc.join()
    ps_proc.terminate()
    logging.info('Work queue job finished.')

  def test_monitored_session_restore(self):
    def save_restore_fn(save_path):
      with ops.Graph().as_default():
        server = server_lib.Server.create_local_server()
        with ops.device('/cpu:0'):
          training_util.get_or_create_global_step()
          work_queue = WorkQueue(
              [b"fast", b"fox", b"jumps", b"over", b"lazy", b"dog"],
              num_epochs=3, shuffle=False, name='wq_save_restore')
          local_queue = work_queue.input_producer()
          dequeue = local_queue.dequeue()
        with monitored_session.MonitoredTrainingSession(
            master=server.target,
            checkpoint_dir=save_path) as sess:
          sess.run(dequeue)
    save_path = self.get_temp_dir()
    save_proc = Process(target=save_restore_fn, args=(save_path,))
    save_proc.start()
    save_proc.join()
    logging.info('work queue saved at {}'.format(save_path))
    restore_proc = Process(target=save_restore_fn, args=(save_path,))
    restore_proc.start()
    restore_proc.join()
    logging.info('work queue restored at {}'.format(save_path))

  def test_monitored_session_change_works_then_restore(self):
    def save_restore_fn(save_path, works):
      with ops.Graph().as_default():
        server = server_lib.Server.create_local_server()
        with ops.device('/cpu:0'):
          training_util.get_or_create_global_step()
          work_queue = WorkQueue(
              works, num_epochs=3, shuffle=False, name='wq_works_changed')
          local_queue = work_queue.input_producer()
          dequeue = local_queue.dequeue()
        with monitored_session.MonitoredTrainingSession(
            master=server.target,
            checkpoint_dir=save_path) as sess:
          sess.run(dequeue)
    save_path = self.get_temp_dir()
    save_proc = Process(
        target=save_restore_fn,
        args=(save_path, [b"to", b"be", b"or", b"not", b"to", b"be"]))
    save_proc.start()
    save_proc.join()
    logging.info('work queue saved at {}'.format(save_path))
    restore_proc = Process(
        target=save_restore_fn,
        args=(save_path, [b"fast", b"fox", b"jumps", b"over", b"lazy", b"dog"]))
    restore_proc.start()
    restore_proc.join()
    logging.info('work queue restored at {}'.format(save_path))

  def test_monitored_session_restore_from_incompatible_checkpoints(self):
    save_path = self.get_temp_dir()
    with ops.Graph().as_default():
      with ops.device('/cpu:0'):
        training_util.get_or_create_global_step()
        vs.get_variable("mock", initializer=array_ops.zeros([10]))
        init = variables.global_variables_initializer()
        incompatible_saver = saver.Saver()
      with self.test_session() as sess:
        sess.run(init)
        # pylint: disable=not-callable
        incompatible_saver.save(sess, os.path.join(save_path, 'test_nockpt'))
    def restore_fn(save_path):
      server = server_lib.Server.create_local_server()
      with ops.Graph().as_default():
        with ops.device('/cpu:0'):
          training_util.get_or_create_global_step()
          work_queue = WorkQueue(
              [b"fast", b"fox", b"jumps", b"over", b"lazy", b"dog"],
              num_epochs=3, name='wq_restore_nothing')
          local_queue = work_queue.input_producer()
          dequeue = local_queue.dequeue()
          default_saver = saver.Saver(
              builder=PartialRestoreSaverBuilder(checkpoint_dir=save_path))
          ops.add_to_collection(ops.GraphKeys.SAVERS, default_saver)
        with monitored_session.MonitoredTrainingSession(
            master=server.target,
            checkpoint_dir=save_path) as sess:
          sess.run(dequeue)
          logging.info('work queue restored at {}'.format(save_path))
    restore_proc = Process(target=restore_fn, args=(save_path,))
    restore_proc.start()
    restore_proc.join()

  def test_simple_dataset(self):
    with self.test_session() as sess:
      works = [b"to", b"be", b"or", b"not", b"to", b"be"]
      num_epochs = 3
      work_queue = WorkQueue(works, num_epochs=num_epochs, shuffle=False)
      dataset = work_queue.input_dataset()
      iterator = dataset.make_initializable_iterator()
      data = iterator.get_next()

      sess.run(iterator.initializer)
      resources.initialize_resources(resources.shared_resources()).run()
      variables.global_variables_initializer().run()
      variables.local_variables_initializer().run()

      threads = queue_runner_impl.start_queue_runners()

      local_works = []
      for _ in range(len(works)*num_epochs):
        local_works.append(sess.run(data))

      self.assertEqual(works * num_epochs, local_works)

      with self.assertRaises(errors_impl.OutOfRangeError):
        sess.run(data)

      for thread in threads:
        thread.join()

  def _get_workers(
      self, num_workers, workers,
      works, num_epochs=1, shuffle=True,
      restore_works_dir=None,
      prefix=None):
    sessions = []
    graphs = []
    train_ops = []
    for worker_id in range(num_workers):
      graph = ops.Graph()
      is_chief = (worker_id == 0)
      with graph.as_default():
        worker_device = "/job:worker/task:%d/cpu:0" % (worker_id)
        with ops.device(device_setter.replica_device_setter(
            worker_device=worker_device,
            ps_device="/job:ps/task:0/cpu:0",
            ps_tasks=1)):

          local_work_mgr = None
          if restore_works_dir:
            local_work_mgr = LocalWorkMgr(
                'worker', worker_id, restore_works_dir)

          work_queue = WorkQueue(
              works, num_epochs=num_epochs, shuffle=shuffle,
              name='global_queue', prefix=prefix,
              local_work_mgr=local_work_mgr)
          dataset = work_queue.input_dataset()
          iterator = dataset.make_initializable_iterator()
          ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS,
                                iterator.initializer)
          data = iterator.get_next()
        # Creates MonitoredSession
        hooks = []
        if local_work_mgr:
          hooks.append(local_work_mgr.hook())
        sess = monitored_session.MonitoredTrainingSession(
            workers[worker_id].target,
            is_chief=is_chief,
            hooks=hooks)

      sessions.append(sess)
      graphs.append(graph)
      train_ops.append(data)

    return sessions, graphs, train_ops

  def test_multi_worker_dataset(self):
    def _run(train_op, sess, result):
      while True:
        try:
          result.append(sess.run(train_op))
        except errors_impl.OutOfRangeError:
          break

    num_ps = 1
    num_workers = 2
    workers, _ = test_util.create_local_cluster(
        num_workers=num_workers, num_ps=num_ps)
    works = [b"fast", b"fox", b"jumps", b"over", b"lazy", b"dog"]
    num_epochs = 3
    sess, _, train_ops = self._get_workers(
        num_workers, workers, works, num_epochs)
    manager = Manager()
    result = manager.list()
    threads = []
    threads.append(
        self.checkedThread(
            target=_run, args=(train_ops[0], sess[0], result)))
    threads.append(
        self.checkedThread(
            target=_run, args=(train_ops[1], sess[1], result)))

    # The two workers starts to execute the train op.
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    self.assertEqual(sorted(works * num_epochs), sorted(result))

    with self.assertRaises(errors_impl.OutOfRangeError):
      sess[0].run(train_ops[0])

    with self.assertRaises(errors_impl.OutOfRangeError):
      sess[1].run(train_ops[1])

  def test_multi_worker_take_for_inference_job(self):
    def _run(train_op, sess, result):
      while True:
        try:
          result.append(sess.run(train_op))
        except errors_impl.OutOfRangeError:
          break
    num_ps = 1
    num_workers = 2
    workers, _ = test_util.create_local_cluster(
        num_workers=num_workers, num_ps=num_ps)
    works = [b"hdfs://tproject/tables/ttable?start=0&end=4",\
             b"hdfs://tproject/tables/ttable?start=4&end=8", \
             b"hdfs://tproject/tables/ttable?start=8&end=12", \
             b"hdfs://tproject/tables/ttable?start=12&end=16"]
    restore_works_dir = '/tmp/workqueue_test_{}'.format(random.randint(50, 99))
    os.system('mkdir -p {}'.format(restore_works_dir))
    sess, _, train_ops = self._get_workers(
        num_workers, workers, works, num_epochs=1,
        shuffle=False, restore_works_dir=restore_works_dir)
    manager = Manager()
    result = manager.list()
    threads = []
    threads.append(
        self.checkedThread(
            target=_run, args=(train_ops[0], sess[0], result)))
    threads.append(
        self.checkedThread(
            target=_run, args=(train_ops[1], sess[1], result)))
    # The two workers starts to execute the train op.
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    restore_works = []
    for home, _, files in os.walk(restore_works_dir):
      for filename in files:
        work_path = os.path.join(home, filename)
        with open(work_path, 'r') as rfile:
          for line in rfile.readlines():
            restore_works.append(line.strip())
    os.system('rm -rf {}'.format(restore_works_dir))

    restore_works = [
        w.encode() if isinstance(w, string_types) else w for w in restore_works]
    self.assertEqual(sorted(works), sorted(restore_works))
    self.assertEqual(sorted(works), sorted(result))

    with self.assertRaises(errors_impl.OutOfRangeError):
      sess[0].run(train_ops[0])
    with self.assertRaises(errors_impl.OutOfRangeError):
      sess[1].run(train_ops[1])

  def test_restore_work_for_inference_job(self):
    def _run(train_op, sess, result):
      while True:
        try:
          result.append(sess.run(train_op))
        except errors_impl.OutOfRangeError:
          break
    num_ps = 1
    num_workers = 1
    workers, _ = test_util.create_local_cluster(
        num_workers=num_workers, num_ps=num_ps)
    restore_work = b"hdfs://tproject/tables/ttable?start=0&end=4"
    works = [b"hdfs://tproject/tables/ttable?start=4&end=8", \
             b"hdfs://tproject/tables/ttable?start=8&end=12", \
             b"hdfs://tproject/tables/ttable?start=12&end=16"]
    restore_works_dir = '/tmp/workqueue_test_{}'.format(random.randint(20, 50))
    os.system('mkdir -p {}/worker_0'.format(restore_works_dir))
    os.system('echo \"{}\" > {}/worker_0/start_0_end_4'.format(
        restore_work.decode(), restore_works_dir))
    sess, _, train_ops = self._get_workers(
        num_workers, workers, works, num_epochs=1,
        shuffle=False, restore_works_dir=restore_works_dir)
    manager = Manager()
    result = manager.list()
    thread = \
        self.checkedThread(target=_run, args=(train_ops[0], sess[0], result))
    thread.start()
    thread.join()
    os.system('rm -rf {}'.format(restore_works_dir))

    result = [
        w.encode() if isinstance(w, string_types) else w for w in result]
    print("result:{}".format(result))
    self.assertEqual([restore_work]+works, list(result))

    with self.assertRaises(errors_impl.OutOfRangeError):
      sess[0].run(train_ops[0])

  def test_restore_work_with_prefix_for_inference_job(self):
    def _run(train_op, sess, result):
      while True:
        try:
          result.append(sess.run(train_op))
        except errors_impl.OutOfRangeError:
          break
    num_ps = 1
    num_workers = 1
    workers, _ = test_util.create_local_cluster(
        num_workers=num_workers, num_ps=num_ps)
    prefix = "hdfs://tproject/tables/"
    restore_work = b"?start=0&end=4"
    works = [b"?start=4&end=8", b"?start=8&end=12", b"?start=12&end=16"]
    restore_works_dir = '/tmp/workqueue_test_{}'.format(random.randint(0, 20))
    os.system('mkdir -p {}/worker_0'.format(restore_works_dir))
    os.system('echo \"{}\" > {}/worker_0/start_0_end_4'.format(
        restore_work.decode(), restore_works_dir))

    sess, _, train_ops = self._get_workers(
        num_workers, workers, works, num_epochs=1,
        shuffle=False, restore_works_dir=restore_works_dir, prefix=prefix)
    manager = Manager()
    result = manager.list()
    thread = \
        self.checkedThread(target=_run, args=(train_ops[0], sess[0], result))
    thread.start()
    thread.join()
    os.system('rm -rf {}'.format(restore_works_dir))

    result = [
        w.encode() if isinstance(w, string_types) else w for w in result]
    print("result:{}".format(result))
    works = [(prefix + str(w.decode())).encode() for w in [restore_work]+ works]
    self.assertEqual(works, list(result))

    with self.assertRaises(errors_impl.OutOfRangeError):
      sess[0].run(train_ops[0])


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  test_lib.main()
