# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Work queue for storing input paths."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from six import string_types
from six.moves import xrange

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_work_queue_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import training

ops.NotDifferentiable('WorkQueueHandleOp')
ops.NotDifferentiable('WorkQueueSave')
ops.NotDifferentiable('WorkQueueCreate')
ops.NotDifferentiable('WorkQueueRestore')
ops.NotDifferentiable('WorkQueueIsInitialized')
ops.NotDifferentiable('WorkQueuePut')
ops.NotDifferentiable('WorkQueueTake')
ops.NotDifferentiable('WorkQueueSize')
ops.NotDifferentiable('WorkQueueClose')
ops.NotDifferentiable('SaveLocalWork')


class Work(object): # pylint: disable=useless-object-inheritance
  """Work with an URL."""
  @classmethod
  def from_url(cls, prefix, url):
    """Creates Work from url.

    Args:
      prefix: Prefix of all works.
      url: Work path.
    """
    if isinstance(url, bytes):
      url = str(url.decode())
    prefix = prefix or ''
    fullpath = str(prefix) + str(url)
    return Work(prefix, url)

  def __init__(self, prefix, url):
    """Initializes the work.

    Args:
      prefix: Prefix of all works.
      url: Work path.
    """
    self._prefix = prefix
    self._url = url

  @property
  def prefix(self):
    """Prefix of all works."""
    return self._prefix

  @property
  def url(self):
    """URL of the work."""
    return self._url

  def count_records(self):
    """Count total number of records."""
    return None

  # pylint: disable=unused-argument
  def get_slice(self, start, end):
    """Get URL of the slice.

    Args:
      start: start index.
      end: end index.
    """
    return None
  # pylint: enable=unused-argument

class WorkQueue(saver.BaseSaverBuilder.SaveableObject):
  """A queue of works shared by all workers.

  A work queue is a queue that shares works for all workers. Any worker can use
  `take` or `input_producer` to take a work from this queue. On initialization,
  this queue will be populated by multiple epochs of work slices. Once failover
  happened, this queue can be restored from latest checkpoint.
  """
  class Resource(object): # pylint: disable=useless-object-inheritance
    """Resource object of a work queue."""
    def __init__(self, name, works):
      self._name = name
      self._works = works

    @property
    def name(self):
      """Resource name of the work queue."""
      return self._name

    @property
    def handle(self):
      """Resource handle of the work queue."""
      return self._works._handle  # pylint: disable=protected-access

    @property
    def create(self):
      """Resource creation op of the work queue."""
      return self._works._create  # pylint: disable=protected-access

    @property
    def is_initialized(self):
      """Resource creation check op of the work queue."""
      return self._works._is_initialized  # pylint: disable=protected-access

  def __init__(
      self,
      works,
      num_epochs=1,
      shuffle=True,
      seed=None,
      prefix=None,
      num_slices=None,
      num_clients=1,
      name=None,
      local_work_mgr=None):
    """Constructs a work queue.

    Args:
      works: A list of input paths.
      num_epochs: (Optional.) An integer. If specified, this work queue
        produces each work from `works` `num_epochs` times before
        generating an `OutOfRange` error. 1 by default.
      shuffle: (Optional.) Boolean. If true, the works are randomly shuffled
        within each epoch.
      seed: (Optional.) An integer. Seed used if shuffle == True.
      prefix: (Optional.) Common prefix of all works.
      num_slices: (Optional.) Total number of slices on all workers.
      num_clients: (Optional.) Number of threads for taking works.
      name: (Optional.) Name of the work queue.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    try:
      executing_eagerly = context.executing_eagerly()
    except: # pylint: disable=bare-except
      executing_eagerly = context.in_eager_mode()
    else:
      executing_eagerly = False
    if not executing_eagerly:
      name = ops.get_default_graph().unique_name(name or 'work_queue')
    else:
      name = name or context.context().scope_name

    if not isinstance(works, list) or not works:
      raise ValueError(
          "WorkQueue requires works as a list of strings")

    works = [
        w.encode() if isinstance(w, string_types) else w for w in works]
    if not all([isinstance(w, bytes) for w in works]):
      raise ValueError(
          "WorkQueue requires works as a list of strings not {}".format(
              [type(w) for w in works]))
    self._works = [w.strip() for w in works]
    self._prefix = prefix
    self._num_clients = num_clients
    self._local_work_mgr = local_work_mgr

    if num_epochs <= 0:
      raise ValueError("num_epochs must be > 0 not {}.".format(num_epochs))

    with ops.name_scope(name):
      self._remote_device = vs.variable(
          0,
          name="colocator",
          trainable=False,
          validate_shape=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES]).device
      self._local_device = control_flow_ops.no_op().device
      with ops.device(self._remote_device):
        self._handle = gen_work_queue_ops.work_queue_handle_op(shared_name=name)
        self._digest_op = ops.convert_to_tensor(
            self.digest, dtype=dtypes.string)
        self._save = gen_work_queue_ops.work_queue_save(self._handle)
        specs = [
            saver.BaseSaverBuilder.SaveSpec(
                self._digest_op, "", name + "_digest"),
            saver.BaseSaverBuilder.SaveSpec(
                self._save, "", name + "_works")]
        slices = []
        if num_slices is not None:
          for work in self._works:
            work_item = Work.from_url(self._prefix, work)
            num_records = work_item.count_records()
            if num_records is None:
              logging.info("[%s] Add work %s .", name, work)
              slices.append(work)
              continue
            if num_slices < 1:
              num_slices = 1
            slice_size = int(num_records / num_slices)
            if slice_size < 1:
              slice_size = 1
            num_slices = int(num_records / slice_size)
            if num_records > num_slices * slice_size:
              num_slices += 1
            logging.info(
                "[%s] Add work %s with %s slices of %s records.",
                name, work, num_slices, num_records)
            for slice_index in xrange(num_slices):
              start = slice_index * slice_size
              end = start + slice_size
              if end > num_records:
                end = num_records
              slices.append(work_item.get_slice(start, end))
        self._capacity = len(slices) if slices else len(self._works)
        works_tensor = ops.convert_to_tensor(
            slices or self._works, dtype=dtypes.string)
        self._create = gen_work_queue_ops.work_queue_create(
            self._handle, shared_name=name)
        for epoch_index in xrange(num_epochs):
          with ops.control_dependencies([self._create]):
            with ops.name_scope('epochs/{}'.format(epoch_index)):
              epoch = works_tensor
              if shuffle:
                epoch = random_ops.random_shuffle(epoch, seed=seed)
              with ops.control_dependencies(
                  [logging_ops.print_v2(
                      "Add epoch of",
                      array_ops.size(epoch),
                      "elements:",
                      epoch,
                      summarize=8)]):
                epoch = array_ops.identity(epoch)
              self._create = gen_work_queue_ops.work_queue_put(self._handle, epoch)
        with ops.control_dependencies([self._create]):
          self._create = gen_work_queue_ops.work_queue_close(self._handle)
        self._is_initialized = gen_work_queue_ops.work_queue_is_initialized(
            self._handle)

    if self._local_work_mgr:
      self._local_work_mgr.get_local_workqueue()

    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, self)
    ops.add_to_collection(
        ops.GraphKeys.RESOURCES, WorkQueue.Resource(name, self))
    logging.info("%s placed at %s.", name, self._remote_device)
    super(WorkQueue, self).__init__(self, specs, name)

  def __len__(self):
    """Number of elements in the work queue."""
    return self._capacity

  @property
  def works(self):
    """The works in the work queue."""
    return self._works

  @property
  def num_clients(self):
    """Number of clients of the work queue."""
    return self._num_clients

  @property
  def digest(self):
    """The digest of works."""
    return b','.join(self._works)

  def load_from_checkpoint(
      self, ckpt_dir_or_file, filename_tensor, preferred_shard):
    """Loads tensors from the checkpoint.
    """
    del preferred_shard

    ckpt_ready = False
    try:
      ckpt_reader = checkpoint_utils.load_checkpoint(ckpt_dir_or_file)
      tensors_in_ckpt = ckpt_reader.get_variable_to_shape_map()
      ckpt_ready = all([spec.name in tensors_in_ckpt for spec in self.specs])
      del tensors_in_ckpt
      del ckpt_reader
    except:  # pylint: disable=bare-except
      pass

    # If tensors found in the checkpoint, do normal restoration.
    if ckpt_ready:
      return [
          io_ops.restore_v2(
              filename_tensor,
              [spec.name],
              [spec.slice_spec],
              [spec.dtype])[0]
          for spec in self.specs]

    # If no tensors found in the checkpoint, just return None.
    return [None, None]

  def restore(self, restored_tensors, _):
    """Restores the work queue from restored_tensors.

    Args:
      restored_tensors: Tensor tuple (digest, works).
    """
    if len(restored_tensors) != 2:
      raise ValueError('WorkQueue requires 2 tensors to restore')
    if restored_tensors[0] is None or restored_tensors[1] is None:
      logging.info("Work queue %s not found in checkpoint.", self.name)
      with ops.name_scope("{}_restore".format(self.name)):
        return self._create
    logging.info("Restore work queue %s.", self.name)
    packed_digest = ops.convert_to_tensor(
        restored_tensors[0], dtype=dtypes.string)
    current_digest = ops.convert_to_tensor(
        self.digest, dtype=dtypes.string)
    same_works_again = math_ops.equal(packed_digest, current_digest)
    works = ops.convert_to_tensor(
        restored_tensors[1], dtype=dtypes.string)
    with ops.control_dependencies([self._create]):
      create_with_prompt = logging_ops.print_v2(
          "Works queue {} abandoned in checkpoint.".format(self.name))
    with ops.name_scope("{}/restore".format(self.name)):
      return control_flow_ops.cond(
          same_works_again,
          lambda: gen_work_queue_ops.work_queue_restore(self._handle, works),
          lambda: create_with_prompt)

  def take(self):
    """Take work from the work queue."""
    def remote_take():
      """Take work from remote worker."""
      with ops.name_scope(self.name):
        with ops.device(self._remote_device):
          taken = gen_work_queue_ops.work_queue_take(
              self._handle,
              num_clients=self.num_clients)

          work_bak = control_flow_ops.no_op()
          if self._local_work_mgr:
            work_bak = gen_work_queue_ops.save_local_work(
                taken,
                job_name=self._local_work_mgr.job_name,
                task_index=self._local_work_mgr.task_index,
                restore_works_dir=self._local_work_mgr.restore_works_dir)
      with ops.control_dependencies([work_bak]):
        with ops.device(self._local_device):
          local_work = array_ops.identity(taken)
          return local_work

    def local_take():
      """Take work from local worker."""
      assert self._local_work_mgr, 'local_work_mgr should not be None.'
      return self._local_work_mgr.take()

    if self._local_work_mgr:
      with ops.name_scope(self.name):
        with ops.device(self._local_device):
          with ops.device('/cpu:0'):
            local_workqueue_empty = self._local_work_mgr.local_workqueue_empty()
            local_work = control_flow_ops.cond(
                local_workqueue_empty, remote_take, local_take)
    else:
      local_work = remote_take()

    if self._prefix is None:
      return local_work
    return string_ops.string_join([self._prefix, local_work])

  def input_producer(self):
    """Returns a FIFOQueue as input producer.

    Returns:
      A local queue of work items.  A `QueueRunner` for the Queue
      is added to the current `Graph`'s `QUEUE_RUNNER` collection.
    """
    work = self.take()
    with ops.name_scope(self.name):
      with ops.device(self._local_device):
        proxy = data_flow_ops.FIFOQueue(
            capacity=1,
            dtypes=[dtypes.string],
            shapes=[tensor_shape.TensorShape([1])],
            name='proxy')
        with ops.control_dependencies(
            [logging_ops.print_v2("Take work:", work)]):
          work = array_ops.identity(work)
        enqueue_proxy = proxy.enqueue(array_ops.reshape(work, (1,)))
        cancel_proxy = proxy.close(cancel_pending_enqueues=True)
        proxy_runner = queue_runner.QueueRunner(
            proxy, [enqueue_proxy], cancel_op=cancel_proxy)
        queue_runner.add_queue_runner(proxy_runner)
        return proxy

  def input_dataset(self):
    """Returns a dataset as input dataset

    Returns:
      A local dataset of work items.
    """
    proxy = self.input_producer()
    next_work = lambda _: array_ops.reshape(proxy.dequeue(), [])
    with ops.name_scope(self.name):
      with ops.device(self._local_device):
        dataset = dataset_ops.Dataset.from_tensors(0).repeat()
        dataset = dataset.map(next_work)
        return dataset

  def add_summary(self):
    """Gets size of the work queue.

    Returns:
      Size of the work queue.
    """
    with ops.name_scope(self.name):
      with ops.device(self._remote_device):
        size = gen_work_queue_ops.work_queue_size(self._handle)
    summary.scalar(
        "{}/fraction_of_{}_full".format(self.name, self._capacity),
        math_ops.to_float(size) * (1. / self._capacity))


class LocalWorkMgr(object):
  """A local work manager for inference job."""
  def __init__(self, job_name, task_index, restore_works_dir, name=None):
    """Constructs a LocalWorkMgr.

    Args:
      job_name: name of current tf-worker.
      task_index: index of current tf-worker.
      restore_works_dir: a directory that restore works for
        WorkQueue when failover.
      name: (Optional.) Name of the LocalWorkMgr.

    Raises:
      ValueError: If one of the arguments is invalid.
    """

    self._job_name = job_name
    self._task_index = task_index
    self._restore_works_dir = restore_works_dir
    self._name = name or 'local_workqueue_mgr'
    assert job_name in ['worker', 'chief'], \
        'support chief or worker role, not {}'.format(job_name)
    assert task_index >= 0, 'task_index must be >= 0, not {}'.format(task_index)
    assert gfile.IsDirectory(restore_works_dir), \
        '{} is not a directory.'.format(restore_works_dir)
    self._get_local_works()
    self._local_device = control_flow_ops.no_op().device

  @property
  def job_name(self):
    return self._job_name

  @property
  def task_index(self):
    return self._task_index

  @property
  def restore_works_dir(self):
    return self._restore_works_dir

  def _get_local_works(self):
    """Get the local works that needs to be restored."""
    self._restore_works = []
    restore_work_file_dir = os.path.join(
        self._restore_works_dir,
        '{}_{}'.format(self._job_name, self._task_index))
    if gfile.IsDirectory(restore_work_file_dir):
      restore_work_files = gfile.ListDirectory(restore_work_file_dir)
      for work_file in restore_work_files:
        work_file_path = os.path.join(restore_work_file_dir, work_file)
        with gfile.GFile(work_file_path, 'r') as rfile:
          for line in rfile.readlines():
            line = line.strip()
            if not re.match(r'(.)*(?:\?start=\d+&end=\d+)$', line):
              logging.error('Invalid format: {}'.format(line))
              continue
            self._restore_works.append(line)
    self._restore_works.sort(
        key=lambda elm: int(re.findall(r'start=(.*)&', elm)[0]))
    logging.info('Restore local worker works:{}'.format(self._restore_works))

  def get_local_workqueue(self):
    """Constructs a local work queue."""
    with ops.name_scope(self._name):
      with ops.device(self._local_device):
        with ops.device('/cpu:0'):
          self._restore_work_queue = data_flow_ops.FIFOQueue(
              capacity=1024,
              dtypes=[dtypes.string],
              shapes=[tensor_shape.TensorShape([1])],
              name='restore_work_queue')
          self.close_restore_work_queue = self._restore_work_queue.close()
          self.restore_work_queue_enqueue = None
          use_barrier = False
          if self._restore_works:
            restore_works_tensors = \
                [array_ops.reshape(work, (1,)) for work in self._restore_works]
            self.restore_work_queue_enqueue = \
                self._restore_work_queue.enqueue_many([restore_works_tensors])
            use_barrier = True
          self.restore_works_barrier = vs.variable(
              use_barrier,
              name="restore_works_barrier",
              trainable=False, validate_shape=False,
              collections=[ops.GraphKeys.LOCAL_VARIABLES])

  def take(self):
    """Take work from the local workqueue."""
    with ops.name_scope(self._name):
      with ops.device(self._local_device):
        with ops.device('/cpu:0'):
          return self._restore_work_queue.dequeue()

  def hook(self):
    """Get a Hook that controls the restore of the works."""
    local_work_mgr = self

    class RestoreWorksBarrierHook(training.SessionRunHook):
      """Hook that controls the restore of the works
         for inference job when failover.
      """
      def __init__(self):
        super(RestoreWorksBarrierHook, self).__init__()
        self._local_work_mgr = local_work_mgr

      def begin(self):
        self._update_barrier = \
            self._local_work_mgr.restore_works_barrier.assign(False)

      def after_create_session(self, session, _):
        if self._local_work_mgr.restore_work_queue_enqueue:
          session.run(self._local_work_mgr.restore_work_queue_enqueue)
          old_barrier = session.run(self._local_work_mgr.restore_works_barrier)
          new_barrier = session.run(self._update_barrier)
          logging.info(
              "Update restore_works_barrier:"
              "{}->{}".format(old_barrier, new_barrier))

      def end(self, session):
        session.run(self._local_work_mgr.close_restore_work_queue)
        logging.info("Close restore_work_queue.")

    return RestoreWorksBarrierHook()

  def local_workqueue_empty(self):
    """Return the truth value if local workqueue empty."""
    with ops.name_scope(self._name):
      with ops.device(self._local_device):
        with ops.device('/cpu:0'):
          def _body(_):
            with ops.control_dependencies([
                logging_ops.print_v2(
                    "Wait to set restore_works_barrier to false,",
                    " maybe LocalWorkMgr hook not set."
                    " restore_works_barrier:", self.restore_works_barrier)]):
              return constant_op.constant(False)
          # wait for the local works restored to complete
          wait = control_flow_ops.while_loop(
              lambda _: self.restore_works_barrier,
              _body,
              [self.restore_works_barrier],
              parallel_iterations=1)
          with ops.control_dependencies([wait]):
            queue_size = self._restore_work_queue.size()
            return math_ops.equal(queue_size, 0)
