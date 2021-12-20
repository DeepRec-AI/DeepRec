import tensorflow as tf

import random

class CriteoClickLogs(object):
  '''Criteo 1TB click logs Dataset.

  See: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
  - 13 dense features taking integer values (mostly count features)
  - 26 sparse features, of which values have been hashed onto 32 bits
    for anonymization purposes.
  '''
  def __init__(
      self, data_dir, batch_size,
      validation=False,
      reader_threads=4,
      reader_buffer=4,
      parser_threads=4,
      parser_buffer=4):
    self.validation = validation
    if validation:
      self.filenames = [
          '{}/val/day_{}_{}.dat'.format(data_dir, self.num_days-1, i)
          for i in range(self.day_splits)]
    else:
      self.filenames = [
          '{}/train/day_{}_{}.dat'.format(data_dir, d, i)
          for d in range(self.num_days)
          for i in range(self.day_splits)]
    self.batch_size = batch_size
    self.reader_threads = reader_threads
    self.reader_buffer = reader_buffer
    self.parser_threads = parser_threads
    self.parser_buffer = parser_buffer
    self.label_name = 'label'
    self.dense_name = 'dense'
    self.sparse_names = [
        'sparse{:02d}'.format(i) for i in range(self.sparse_dims)]
    self.mask_names = [
        'mask{:02d}'.format(i) for i in range(self.sparse_dims)]

  @property
  def num_days(self):
    return 1 

  @property
  def day_splits(self):
    return 1

  @property
  def record_bytes(self):
    return 160

  @property
  def label_dims(self):
    return 1

  @property
  def dense_dims(self):
    return 13

  @property
  def sparse_dims(self):
    return 26

  @property
  def sparse_bucket_sizes(self):
    return [
        39884406,
        39043,
        17289,
        7420,
        20263,
        3,
        7120,
        1543,
        63,
        38532951,
        2953546,
        403346,
        10,
        2208,
        11938,
        155,
        4,
        976,
        14,
        39979771,
        25641295,
        39664984,
        585935,
        12972,
        108,
        36]

  @property
  def dims(self):
    return self.label_dims + self.dense_dims + self.sparse_dims

  def _partition(self, shard, num_shards):
    all_works = []
    for fname in self.filenames:
      with tf.gfile.GFile(fname) as f:
        total_bytes = f.size()
      batch_bytes = self.record_bytes * self.batch_size
      extra_bytes = total_bytes % batch_bytes
      num_batches = (total_bytes - extra_bytes) // batch_bytes
      num_readers = self.reader_threads * num_shards
      work_sizes = [num_batches // num_readers for i in range(num_readers)]
      work_offsets = [0]
      works = [(fname, 0, total_bytes - work_sizes[0] * batch_bytes)]
      for i in range(1, num_readers):
        work_offsets.append(work_offsets[i-1] + work_sizes[i-1])
        works.append((
            fname,
            work_offsets[i] * batch_bytes,
            total_bytes - (work_offsets[i] + work_sizes[i]) * batch_bytes))
      all_works.extend(works)
    all_works = all_works[shard::num_shards]
    random.shuffle(all_works)
    return tuple(tf.convert_to_tensor(t) for t in zip(*all_works))

  def _make_dataset(self, name, head=0, foot=0):
    return tf.data.FixedLengthRecordDataset(
        name,
        tf.to_int64(self.record_bytes * self.batch_size),
        tf.to_int64(head),
        tf.to_int64(foot))

  def _next(self, batch):
    record = tf.reshape(tf.io.decode_raw(batch, tf.int32), [-1, self.dims])
    label, dense, sparse = tf.split(
        record,
        [self.label_dims, self.dense_dims, self.sparse_dims], 1)
    label = tf.to_float(label)
    dense = tf.log(tf.to_float(dense) + 1.)
    sparse_slices = tf.unstack(sparse, axis=1)
    feats = {self.label_name: label, self.dense_name: dense}
    for sidx in range(self.sparse_dims):
      feats[self.sparse_names[sidx]] = tf.floormod(
          sparse_slices[sidx], self.sparse_bucket_sizes[sidx])
      feats[self.mask_names[sidx]] = tf.split(
          tf.to_float(tf.not_equal(sparse, -1)),
          self.sparse_dims, 1)
    return feats

  def as_dataset(self, shard=0, num_shards=1):
    works = tf.data.Dataset.from_tensor_slices(self._partition(shard, num_shards))
    work_reader = tf.data.experimental.parallel_interleave(
        self._make_dataset,
        cycle_length=self.reader_threads,
        block_length=self.reader_buffer,
        sloppy=True)
    ds = works.apply(work_reader)
    return ds.map(self._next, self.parser_threads).prefetch(self.parser_buffer)
