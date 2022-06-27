import numpy as np
import pandas as pd
import os
import time

import tensorflow as tf
from tensorflow.python.client import timeline
from criteo import CriteoClickLogs
from tensorflow.core.framework.embedding import config_pb2
from tensorflow.python.ops import variables

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("batch_size", 1280, "")
tf.app.flags.DEFINE_integer("num_steps", 1000, "")
tf.app.flags.DEFINE_integer("dim_size", 256, "")
tf.app.flags.DEFINE_float("lr", 0.1, "")
tf.app.flags.DEFINE_float("l2", 0.0001, "")
tf.app.flags.DEFINE_string("data_dir", '/workspace/criteo', "")
tf.app.flags.DEFINE_boolean("use_mock_data", True, "")
tf.app.flags.DEFINE_integer("num_mock_cols", 100, "")
tf.app.flags.DEFINE_integer("max_mock_id_amplify", 1000, "")
tf.app.flags.DEFINE_integer("mock_vocabulary_size", 10000, "")
tf.app.flags.DEFINE_string("ps_hosts", None, "")
tf.app.flags.DEFINE_string("worker_hosts", '127.0.0.1:8868', "")
tf.app.flags.DEFINE_string("job_name", 'worker', "")
tf.app.flags.DEFINE_integer("task_index", 0, "")
tf.app.flags.DEFINE_integer("vocabulary_amplify_factor", 1, "")
tf.app.flags.DEFINE_boolean("use_ev_var", True, "")
tf.app.flags.DEFINE_boolean("use_xdl_var", False, "")
tf.app.flags.DEFINE_boolean("trace_timeline", False, "")
tf.app.flags.DEFINE_string("ev_storage", 'dram', "")
tf.app.flags.DEFINE_string("ev_storage_path",
                           '/mnt/pmem0/pmem_allocator/', "")
tf.app.flags.DEFINE_integer("ev_storage_size_gb", '512', "")

def main(_):
  cluster_dict = {}
  if FLAGS.ps_hosts is not None:
    cluster_dict['ps'] = FLAGS.ps_hosts.split(',')
  cluster_dict['worker'] = FLAGS.worker_hosts.split(',')
  cluster_spec = tf.train.ClusterSpec(cluster_dict)
  num_workers = len(cluster_dict['worker'])
  is_chief = FLAGS.task_index == 0
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True
  config.gpu_options.force_gpu_compatible = True
  server = tf.train.Server(
      cluster_spec,
      job_name=FLAGS.job_name,
      task_index=FLAGS.task_index,
      config=config)
  if FLAGS.job_name == "ps":
    server.join()
    return

  shard = FLAGS.task_index
  num_shards = num_workers

  if FLAGS.use_mock_data:
    # use mock data
    if FLAGS.use_ev_var or FLAGS.use_xdl_var:
      ## set up a ratio to 
      mock_data = pd.DataFrame(
          np.random.randint(
              0, FLAGS.batch_size * FLAGS.max_mock_id_amplify,
              size=(FLAGS.batch_size * FLAGS.num_steps, FLAGS.num_mock_cols),
              dtype=np.int64),
          columns=['col%d' % c for c in range(FLAGS.num_mock_cols)])
    else:
      mock_data = pd.DataFrame(
          np.random.randint(
              0, FLAGS.mock_vocabulary_size,
              size=(FLAGS.batch_size * 100, FLAGS.num_mock_cols),
              dtype=np.int64),
          columns=['col%d' % c for c in range(FLAGS.num_mock_cols)])
  else:
    click_logs = CriteoClickLogs(
        FLAGS.data_dir, FLAGS.batch_size)
  with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
    with tf.name_scope('io'):
      if FLAGS.use_mock_data: 
        ds = tf.data.Dataset.from_tensor_slices(dict(mock_data)).\
            shuffle(buffer_size = 10 * FLAGS.batch_size).\
            repeat().batch(FLAGS.batch_size).prefetch(1)
        batch = ds.make_one_shot_iterator().get_next()
        features = {'fm_w': [], 'fm_v': []}
      else:
        ds = click_logs.as_dataset(shard, num_shards).repeat().prefetch(1)
        batch = ds.make_one_shot_iterator().get_next()
        features = {
            'label': batch['label'],
            'dense': batch['dense'],
            'fm_w': [],
            'fm_v': []}
    with tf.name_scope('fm'):
      if FLAGS.use_mock_data:
        for sidx in range(FLAGS.num_mock_cols):
          if FLAGS.use_ev_var:
            if FLAGS.ev_storage == "dram":
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.DRAM))
            elif FLAGS.ev_storage == "pmem_memkind":
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(storage_type=config_pb2.StorageType.PMEM_MEMKIND))
            elif FLAGS.ev_storage == "pmem_libpmem":
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(
                                                              storage_type=config_pb2.StorageType.PMEM_LIBPMEM, 
                                                              storage_path=FLAGS.ev_storage_path, 
                                                              storage_size=[FLAGS.ev_storage_size_gb * 1024 * 1024 * 1024]))
            elif FLAGS.ev_storage == "dram_pmem":
              ev_option = variables.EmbeddingVariableOption(storage_option=variables.StorageOption(
                                                              storage_type=config_pb2.StorageType.DRAM_PMEM, 
                                                              storage_path=FLAGS.ev_storage_path, 
                                                              storage_size=[FLAGS.ev_storage_size_gb * 1024 * 1024 * 1024, FLAGS.ev_storage_size_gb * 1024 * 1024 * 1024]))
            fm_w = tf.get_embedding_variable(
               name='fm_w{}'.format(sidx),
               embedding_dim=1,
               key_dtype=tf.int64,
               initializer=tf.ones_initializer(tf.float32),
               ev_option = ev_option)
            features['fm_w'].append(
                tf.nn.embedding_lookup(fm_w, batch['col{}'.format(sidx)]))
            fm_v = tf.get_embedding_variable(
               name='fm_v{}'.format(sidx),
               embedding_dim=FLAGS.dim_size,
               key_dtype=tf.int64,
               initializer=tf.ones_initializer(tf.float32),
               ev_option = ev_option)
            features['fm_v'].append(
                tf.nn.embedding_lookup(fm_v, batch['col{}'.format(sidx)]))
          elif FLAGS.use_xdl_var:
            fm_w = tf.hash_table.DistributedHashTable(
               shape=[1],
               dtype=tf.float32,
               initializer=tf.zeros_initializer(tf.float32),
               partitioner=tf.hash_table.FixedSizeHashTablePartitioner(1),
               name='fm_w{}'.format(sidx))
            features['fm_w'].append(
                fm_w.lookup(batch['col{}'.format(sidx)]))
            fm_v = tf.hash_table.DistributedHashTable(
               shape=[FLAGS.dim_size],
               dtype=tf.float32,
               initializer=tf.zeros_initializer(tf.float32),
               partitioner=tf.hash_table.FixedSizeHashTablePartitioner(1),
               name='fm_v{}'.format(sidx))
            features['fm_v'].append(
                fm_v.lookup(batch['col{}'.format(sidx)]))
          else:
            fm_w = tf.get_variable(
               name='fm_w{}'.format(sidx),
               shape=[FLAGS.mock_vocabulary_size, 1],
               initializer=tf.truncated_normal_initializer(stddev=0.001))         
            features['fm_w'].append(
                tf.nn.embedding_lookup(fm_w, batch['col{}'.format(sidx)]))
            fm_v = tf.get_variable(
                name='fm_v{}'.format(sidx),
                shape=[FLAGS.mock_vocabulary_size, FLAGS.dim_size],
                initializer=tf.truncated_normal_initializer(stddev=0.001))
            features['fm_v'].append(
                tf.nn.embedding_lookup(fm_v, batch['col{}'.format(sidx)]))
      else:
        sparse_names = click_logs.sparse_names
        sparse_bucket_sizes = [x * FLAGS.vocabulary_amplify_factor \
            for x in click_logs.sparse_bucket_sizes]
        for sidx, sname in enumerate(sparse_names):
          fm_w = tf.get_variable(
              name='fm_w{}'.format(sidx),
              shape=[sparse_bucket_sizes[sidx], 1],
              initializer=tf.truncated_normal_initializer(stddev=0.001))
          features['fm_w'].append(tf.nn.embedding_lookup(fm_w, batch[sname]))
          fm_v = tf.get_variable(
              name='fm_v{}'.format(sidx),
              shape=[sparse_bucket_sizes[sidx], FLAGS.dim_size],
              initializer=tf.truncated_normal_initializer(stddev=0.001))
          features['fm_v'].append(tf.nn.embedding_lookup(fm_v, batch[sname]))
    fm_w_features = tf.concat(features['fm_w'], axis=-1)
    fm_v_features = tf.concat(features['fm_v'], axis=-1)
    loss = FLAGS.l2 * (tf.nn.l2_loss(fm_w_features) + \
        tf.nn.l2_loss(fm_v_features))   
    opt = tf.train.AdagradOptimizer(learning_rate=FLAGS.lr)
    step = tf.train.create_global_step()
    train_op = opt.minimize(loss, global_step=step)
  # calculate embedding variable size
  ev_list = tf.get_collection(tf.GraphKeys.EMBEDDING_VARIABLES)
  total_size=4*tf.add_n([tf.divide(tf.reduce_prod(ev.total_count()), 1024*1024) for ev in ev_list])

  hooks = []
  hooks.append(tf.train.LoggingTensorHook({'step': step}, every_n_iter=10))
  if FLAGS.trace_timeline:
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
  with tf.train.MonitoredTrainingSession(
      server.target, is_chief=is_chief, hooks=hooks) as sess:
    durs = []
    prev_ts = time.time()
    for i in range(FLAGS.num_steps):
      if FLAGS.trace_timeline and i % 100 == 0:
        sess.run(train_op, options=options, run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_bench_step_%d.json' % i, 'w') as f:
          f.write(chrome_trace)
      else:
        sess.run(train_op)
      ts = time.time()
      durs.append(ts - prev_ts)
      prev_ts = ts
    total_size=sess.run(total_size)
    durs = np.array(durs)
    tf.logging.info(
        '{} x {} samples with dim {} trained in {:.2f}ms, {:.2f} samples/sec, '
        '(avg={:.2f}ms, p10={:.2f}ms, p50={:.2f}ms, p90={:.2f}ms, '
        'p95={:.2f}ms), ev_mem_request={:.2f} MB.'.format(
            FLAGS.num_steps,
            FLAGS.batch_size,
            FLAGS.dim_size,
            1000 * np.sum(durs),
            FLAGS.batch_size / float(np.mean(durs)),
            1000 * np.mean(durs),
            1000 * np.percentile(durs, 10),
            1000 * np.percentile(durs, 50),
            1000 * np.percentile(durs, 90),
            1000 * np.percentile(durs, 95),
            total_size))

if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

