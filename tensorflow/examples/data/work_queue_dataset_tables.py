# -*- coding: utf-8 -*-
"""Example for work queue dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
from tensorflow.python.ops.work_queue import WorkQueue

tf.app.flags.DEFINE_string(
    'ps_hosts', '', 'must be endpoint list')
tf.app.flags.DEFINE_string(
    'worker_hosts', '127.0.0.1:0', 'must be endpoint list')
tf.app.flags.DEFINE_string(
    'job_name', 'worker', 'must be in ("", "worker", "ps")')
tf.app.flags.DEFINE_integer('task_index', 0, 'must be integer')
tf.app.flags.DEFINE_integer('num_epochs', 3, 'must be integer')
tf.app.flags.DEFINE_integer('save_checkpoint_secs', 10, 'must be integer')
tf.app.flags.DEFINE_string(
    'tables', '', 'must be input path list')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', None, 'Directory for checkpoints')

FLAGS = tf.flags.FLAGS

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster_dict = {"worker": worker_hosts}
  if ps_hosts:
    cluster_dict["ps"] = ps_hosts
  cluster = tf.train.ClusterSpec(cluster_dict)
  server = tf.train.Server(
      cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()
    return

  worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
  works = FLAGS.tables.split(",")
  if not works:
    raise ValueError("--works must be set.")

  with tf.device(tf.train.replica_device_setter(
      worker_device=worker_device, cluster=cluster)):
    global_step = tf.train.get_or_create_global_step()
    work_queue = WorkQueue(
        works, num_epochs=FLAGS.num_epochs, num_slices=len(worker_hosts) * 10)
    filenames_ds = work_queue.input_dataset()
    input_dataset = tf.data.TableRecordDataset(filenames_ds,
                                               record_defaults=('',))
    iterator = input_dataset.make_initializable_iterator()
    values = iterator.get_next()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    with tf.control_dependencies(list(values)):
      train_op = global_step.assign_add(1)

  hooks = []
  is_chief = (FLAGS.task_index == 0)
  with tf.train.MonitoredTrainingSession(
      master=server.target,
      is_chief=is_chief,
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=FLAGS.save_checkpoint_secs,
      hooks=hooks) as sess:

    tf.logging.info('Training starts.')
    count = 0
    while not sess.should_stop():
      try:
        sess.run(train_op)
        count += 1
        time.sleep(0.5)
      except tf.errors.OutOfRangeError:
        break
    tf.logging.info('Training ends for %s reads.', count)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
