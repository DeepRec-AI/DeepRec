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
"""Tests for prefetching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib

from tensorflow.python.ops import prefetch


# pylint: disable=missing-docstring
class PrefetchTest(test.TestCase):
  def test_simple(self):
    capacity = 2
    value = 42.0
    with ops.Graph().as_default() as graph:
      with ops.device('/cpu:0'):
        x = array_ops.constant(value, dtype=dtypes.float32, shape=[])
      with ops.device(test.gpu_device_name()):
        y = prefetch.staged(x, capacity=capacity, num_threads=2, timeout_millis=1000)

    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph) as sess:
      coord = coordinator.Coordinator()
      prefetch.make_prefetch_hook().create_threads(sess, coord)
      for _ in xrange(capacity * 3):
        self.assertAllClose(value, sess.run(y), rtol=1e-6)
      coord.request_stop()

  def test_string(self):
    capacity = 3
    value = "'The quick brown fox jumps over the lazy dog!'"
    with ops.Graph().as_default() as graph:
      with ops.device('/cpu:0'):
        x = array_ops.constant(value, dtype=dtypes.string, shape=[])
      with ops.device(test.gpu_device_name()):
        y = prefetch.staged(x, capacity=capacity, num_threads=6, timeout_millis=1000)

    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph) as sess:
      coord = coordinator.Coordinator()
      prefetch.make_prefetch_hook().create_threads(sess, coord)
      for _ in xrange(capacity * 3):
        self.assertEqual(value, sess.run(y).decode())
      coord.request_stop()

  def test_sparse(self):
    with ops.Graph().as_default() as graph:
      with ops.device('/cpu:0'):
        values = array_ops.constant([1, 1, 1], dtype=dtypes.int64)
        indices = array_ops.constant(
            ([0, 0], [0, 1], [0, 2]), dtype=dtypes.int64)
        dense_shape = array_ops.constant([3, 3], dtype=dtypes.int64)

        x = sparse_tensor.SparseTensor(values=values,
                                       indices=indices,
                                       dense_shape=dense_shape)
      with ops.device(test.gpu_device_name()):
        y = prefetch.staged(x, timeout_millis=1000)

    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph) as sess:
      values_data = sess.run(values)
      indices_data = sess.run(indices)
      dense_shape_data = sess.run(dense_shape)
      coord = coordinator.Coordinator()
      prefetch.make_prefetch_hook().create_threads(sess, coord)
      for _ in xrange(3):
        prefetched = sess.run(y)
        self.assertAllClose(values_data, prefetched.values, rtol=1e-6)
        self.assertAllClose(indices_data, prefetched.indices, rtol=1e-6)
        self.assertAllClose(dense_shape_data, prefetched.dense_shape, rtol=1e-6)
      coord.request_stop()

  def test_list(self):
    with ops.Graph().as_default() as graph:
      with ops.device('/cpu:0'):
        values = array_ops.constant([1, 1, 1], dtype=dtypes.int64)
        indices = array_ops.constant(
            ([0, 0], [0, 1], [0, 2]), dtype=dtypes.int64)
        dense_shape = array_ops.constant([3, 3], dtype=dtypes.int64)

        x1 = sparse_tensor.SparseTensor(values=values,
                                        indices=indices,
                                        dense_shape=dense_shape)
        x2 = array_ops.constant(42.0, dtype=dtypes.float32, shape=[])
        x = [x1, x2]
      with ops.device(test.gpu_device_name()):
        y = prefetch.staged(x, timeout_millis=1000)

    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph) as sess:
      values_data = sess.run(values)
      indices_data = sess.run(indices)
      dense_shape_data = sess.run(dense_shape)
      x2_data = sess.run(x2)
      coord = coordinator.Coordinator()
      prefetch.make_prefetch_hook().create_threads(sess, coord)
      for _ in xrange(3):
        prefetched = sess.run(y)
        self.assertAllClose(values_data, prefetched[0].values, rtol=1e-6)
        self.assertAllClose(indices_data, prefetched[0].indices, rtol=1e-6)
        self.assertAllClose(
            dense_shape_data, prefetched[0].dense_shape,
            rtol=1e-6)
        self.assertAllClose(x2_data, prefetched[1], rtol=1e-6)
      coord.request_stop()

  def test_dict(self):
    with ops.Graph().as_default() as graph:
      with ops.device('/cpu:0'):
        values = array_ops.constant([1, 1, 1], dtype=dtypes.int64)
        indices = array_ops.constant(
            ([0, 0], [0, 1], [0, 2]), dtype=dtypes.int64)
        dense_shape = array_ops.constant([3, 3], dtype=dtypes.int64)

        x1 = sparse_tensor.SparseTensor(values=values,
                                        indices=indices,
                                        dense_shape=dense_shape)
        x2 = array_ops.constant(42.0, dtype=dtypes.float32, shape=[])
        x = {'foo': x2, 'bar': x1}
      with ops.device(test.gpu_device_name()):
        y = prefetch.staged(x, timeout_millis=1000)

    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph) as sess:
      values_data = sess.run(values)
      indices_data = sess.run(indices)
      dense_shape_data = sess.run(dense_shape)
      x2_data = sess.run(x2)
      coord = coordinator.Coordinator()
      prefetch.make_prefetch_hook().create_threads(sess, coord)
      for _ in xrange(3):
        prefetched = sess.run(y)
        self.assertAllClose(values_data, prefetched['bar'].values, rtol=1e-6)
        self.assertAllClose(indices_data, prefetched['bar'].indices, rtol=1e-6)
        self.assertAllClose(
            dense_shape_data, prefetched['bar'].dense_shape, rtol=1e-6)
        self.assertAllClose(x2_data, prefetched['foo'], rtol=1e-6)
      coord.request_stop()

  def test_dict_from_feeds(self):
    def my_generator_fn3(_):
      for i in xrange(3):
        yield [i]

    with ops.Graph().as_default() as graph:
      with ops.device('/cpu:0'):
        values = array_ops.constant([1, 1, 1], dtype=dtypes.int64)
        indices = array_ops.constant(
            ([0, 0], [0, 1], [0, 2]), dtype=dtypes.int64)
        dense_shape = array_ops.constant([3, 3], dtype=dtypes.int64)

        x1 = sparse_tensor.SparseTensor(values=values,
                                        indices=indices,
                                        dense_shape=dense_shape)
        x2 = array_ops.constant(42.0, dtype=dtypes.float32, shape=[])
        x3 = array_ops.placeholder(dtypes.int32, shape=[])
        x = {'foo': x2, 'bar': x1, 'foobar': x3}
      with ops.device(test.gpu_device_name()):
        y = prefetch.staged(
            x, feed_list=[x3], feed_generator=my_generator_fn3,
            timeout_millis=1000)

    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph) as sess:
      values_data = sess.run(values)
      indices_data = sess.run(indices)
      dense_shape_data = sess.run(dense_shape)
      x2_data = sess.run(x2)
      coord = coordinator.Coordinator()
      prefetch.make_prefetch_hook().create_threads(sess, coord)
      for i in xrange(3):
        prefetched = sess.run(y)
        self.assertAllClose(values_data, prefetched['bar'].values, rtol=1e-6)
        self.assertAllClose(indices_data, prefetched['bar'].indices, rtol=1e-6)
        self.assertAllClose(
            dense_shape_data, prefetched['bar'].dense_shape, rtol=1e-6)
        self.assertAllClose(x2_data, prefetched['foo'], rtol=1e-6)
        self.assertAllClose(i, prefetched['foobar'], rtol=1e-6)
      coord.request_stop()

  def test_dict_from_feeds_with_session_run(self):
    with ops.Graph().as_default() as graph:
      ph = array_ops.placeholder(dtypes.int32, shape=[])
      count_op = array_ops.constant(100) + ph
      def my_generator_fn100(sess):
        for i in xrange(100):
          yield [sess.run(count_op, feed_dict={ph: i})]

      with ops.device('/cpu:0'):
        x3 = array_ops.placeholder(dtypes.int32, shape=[])
      with ops.device(test.gpu_device_name()):
        yy = prefetch.staged(
            x3,
            feed_list=[x3],
            feed_generator=my_generator_fn100,
            capacity=4,
            num_threads=4,
            timeout_millis=1000)

    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph) as sess:
      coord = coordinator.Coordinator()
      prefetch.make_prefetch_hook().create_threads(sess, coord)
      for i in xrange(99):
        print(i, ':', sess.run(yy), end=', ')
      print('done.')
      coord.request_stop()

  def test_corrupted_inputs(self):
    def csv_generator(_):
      for i in xrange(10):
        if i < 9:
          yield [u'abc,def']
        else:
          yield [u'corrupted"record,xyz']

    with ops.Graph().as_default() as graph:
      with ops.device('/cpu:0'):
        x1 = array_ops.placeholder(dtypes.string, shape=[])
        x2 = array_ops.constant(42.0, dtype=dtypes.float32, shape=[])
        decoded_x1 = parsing_ops.decode_csv(x1, record_defaults=[[''], ['']], use_quote_delim=False)
        x = {'x1': decoded_x1, 'x2': x2}
        y = prefetch.staged(
            x,
            feed_list=[x1],
            feed_generator=csv_generator,
            ignored_exception_types=(errors.InvalidArgumentError,),
            timeout_millis=1000)

    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph) as sess:
      x2_data = sess.run(x2)
      coord = coordinator.Coordinator()
      prefetch.make_prefetch_hook().create_threads(sess, coord)
      for _ in xrange(9):
        try:
          prefetched = sess.run(y)
        except errors.OutOfRangeError:
          break
        self.assertEqual(
            [u'abc', u'def'],
            [s.decode() for s in prefetched['x1']])
        self.assertAllClose(x2_data, prefetched['x2'], rtol=1e-6)
      try:
        prefetched = sess.run(y)
      except errors.OutOfRangeError:
        pass
      coord.request_stop()

  def test_preemption_retry(self):
    server = server_lib.Server.create_local_server()
    capacity = 5
    value = "'The quick brown fox jumps over the lazy dog!'"
    with ops.Graph().as_default():
      with ops.device('/cpu:0'):
        x = array_ops.constant(value, dtype=dtypes.string, shape=[])
        y = prefetch.staged(x, capacity=capacity, num_threads=3, timeout_millis=1000)

      sess = monitored_session.MonitoredTrainingSession(
          master=server.target,
          hooks=[prefetch.make_prefetch_hook()])
      sess._sess._sess._coord.request_stop() # pylint: disable=protected-access
      sess._sess.close() # pylint: disable=protected-access
      sess._sess._sess = None # pylint: disable=protected-access
      sess._sess.close() # pylint: disable=protected-access
      sess._sess._sess = None # pylint: disable=protected-access
      sess._sess.close() # pylint: disable=protected-access
      sess._sess._sess = None # pylint: disable=protected-access
      sess.run(y)
      sess._sess._sess = None # pylint: disable=protected-access
      sess._sess.close() # pylint: disable=protected-access
      sess.run(y)
      sess.run(y)
      sess.close()

# pylint: enable=missing-docstring

if __name__ == '__main__':
  test.main()
