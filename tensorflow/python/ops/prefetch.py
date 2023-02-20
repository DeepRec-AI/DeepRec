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
"""Prefetching samples asynchronously."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


from tensorflow.python.ops import gen_tensor_buffer_ops
from tensorflow.python.ops.prefetch_runner import PrefetchRunner

ops.NotDifferentiable('TensorBufferPut')
ops.NotDifferentiable('TensorBufferTake')
ops.NotDifferentiable('TensorBufferCancel')

PREFETCH = "prefetch"

@tf_export(v1=["make_prefetch_hook"])
def make_prefetch_hook(daemon=True, start=True):
  """Create PrefetchRunner.Hook for prefetching.

  Args:
    daemon: (Optional.) Whether the threads should be marked as `daemons`,
      meaning they don't block program exit.
    start: (Optional.) If `False` threads would not be started.

  Returns:
    A PrefetchRunner.Hook for prefetching.
  """
  return PrefetchRunner.Hook(PREFETCH, daemon=daemon, start=start)

@tf_export(v1=["staged"])
def staged(
    features,
    feed_list=None,
    feed_generator=None,
    capacity=1,
    num_threads=1,
    num_clients=1,
    timeout_millis=300000,
    closed_exception_types=None,
    ignored_exception_types=None,
    use_stage_subgraph_thread_pool=False,
    stage_subgraph_thread_pool_id = 0,
    stage_subgraph_stream_id = 0,
    name=None):
  """Prefetch samples.

  Args:
    features: Nest structure of tensors to prefetch.
    feed_list: (Optional.) A list of `feed_dict` keys. See
      @{tf.Session.run} for details of the allowable feed key types.
    feed_generator: (Optional.) A generator function lambda sess: iterator
      that yields a list of `feed_dict` values.
    capacity: (Optional.) Max number of samples to keep in the buffer.
    num_threads: (Optional.) Number of threads for prefetching. 1 by
      default.
    num_clients: (Optional.) Number of clients of prefetched sample. 1 by
      default.
    timeout_millis: (Optional.) Max milliseconds put op can take, 5 min by
      default.
    closed_exception_types: (Optional.) Exception types indicating that the
      prefetching is normally finished. Defaults to
      `(tf.errors.OutOfRangeError, StopIteration)`.
    ignored_exception_types: (Optional.) Exception types indicating that the
      prefetching can continue. Defaults to `()`.
    use_stage_subgraph_thread_pool: (Optional.) Use stage subgraph thread pool
      to run stage graph or not.
    stage_subgraph_thread_pool_id: (Optional.) Specifies the stage subgraph
      thread pool to use when enable use_stage_subgraph_thread_pool. 0 by default.
    stage_subgraph_stream_id: (Optional.) Specifies which stream to use for the
      Stage subgraph. The default value is 0.
    name: (Optional.) Name of prefetching operations.

  Returns:
    Prefetched sample.
  """
  if num_threads < 1:
    raise ValueError('num_threads must >= 1')

  if name is None:
    name = ops.get_default_graph().unique_name(PREFETCH)
  with ops.name_scope(name):
    local_device = control_flow_ops.no_op().device
  tensor_or_sparse_tensor_or_nones = nest.flatten(features)

  tensor_or_nones = []
  for t in tensor_or_sparse_tensor_or_nones:
    if hasattr(t, 'dense_shape'):
      tensor_or_nones.extend([t.values, t.indices, t.dense_shape])
    else:
      tensor_or_nones.append(t)

  tensor_indices = []
  tensors = []
  for i, v in enumerate(tensor_or_nones):
    if v is not None:
      tensor_indices.append(i)
      tensors.append(v)
  tensor_dtypes = []
  tensor_shapes = []
  for v in tensors:
    tensor_dtypes.append(v.dtype)
    tensor_shapes.append(v.shape if hasattr(v, 'shape') else None)

  with ops.name_scope(name):
    with ops.device(local_device):
      # only set stream id when stage_subgraph_id > 0,
      # because stream 0 is used by default.
      if (stage_subgraph_stream_id > 0):
        with ops.stream(stage_subgraph_stream_id):
          fetch_tensors = gen_tensor_buffer_ops.tensor_buffer_put(
            tensors,
            timeout_millis=timeout_millis,
            shared_name=name,
            shared_capacity=capacity)
      else:
        fetch_tensors = gen_tensor_buffer_ops.tensor_buffer_put(
          tensors,
          timeout_millis=timeout_millis,
          shared_name=name,
          shared_capacity=capacity)

      cancel_fetching = gen_tensor_buffer_ops.tensor_buffer_cancel(
          shared_name=name,
          shared_capacity=capacity)
      resume_fetching = gen_tensor_buffer_ops.tensor_buffer_cancel(
          is_cancelled=False,
          shared_name=name,
          shared_capacity=capacity)
      close_fetching = gen_tensor_buffer_ops.tensor_buffer_close(
          shared_name=name,
          shared_capacity=capacity)
      next_tensors = gen_tensor_buffer_ops.tensor_buffer_take(
          dtypes=tensor_dtypes,
          shared_name=name,
          shared_capacity=capacity,
          shared_threads=num_clients)
      if not isinstance(next_tensors, (tuple, list)):
        next_tensors = [next_tensors]
      next_tensors = [array_ops.identity(t) for t in next_tensors]
    for i, t in enumerate(next_tensors):
      t.set_shape(tensor_shapes[i])
    next_tensor_or_nones = [None] * len(tensor_or_nones)
    for i, v in enumerate(next_tensors):
      next_tensor_or_nones[tensor_indices[i]] = v
    next_tensor_or_nones = collections.deque(next_tensor_or_nones)
    next_tensor_or_sparse_tensor_or_nones = []
    for t in tensor_or_sparse_tensor_or_nones:
      if hasattr(t, 'dense_shape'):
        sparse_values = next_tensor_or_nones.popleft()
        sparse_indices = next_tensor_or_nones.popleft()
        sparse_dense_shape = next_tensor_or_nones.popleft()
        next_tensor_or_sparse_tensor_or_nones.append(
            sparse_tensor.SparseTensor(
                values=sparse_values,
                indices=sparse_indices,
                dense_shape=sparse_dense_shape))
      else:
        next_tensor_or_sparse_tensor_or_nones.append(
            next_tensor_or_nones.popleft())
    prefetched = nest.pack_sequence_as(
        features, next_tensor_or_sparse_tensor_or_nones)
  runner = PrefetchRunner(
      fetch_ops=[fetch_tensors] * num_threads,
      cancel_op=cancel_fetching,
      resume_op=resume_fetching,
      close_op=close_fetching,
      feed_list=feed_list,
      feed_generator=feed_generator,
      closed_exception_types=closed_exception_types,
      ignored_exception_types=ignored_exception_types,
      use_stage_subgraph_thread_pool=use_stage_subgraph_thread_pool,
      stage_subgraph_thread_pool_id=stage_subgraph_thread_pool_id)
  ops.add_to_collection(PREFETCH, runner)
  return prefetched

@tf_export(v1=["prefetch_join"])
def prefetch_join(
    thread_to_features,
    feed_list=None,
    feed_generator=None,
    capacity=1,
    num_clients=1,
    timeout_millis=300000,
    closed_exception_types=None,
    ignored_exception_types=None,
    name=None):
  """Prefetch samples from thread_to_features list.

  `Unlike `prefetch`, `prefetch_join` runs different ops in different threads.
  `prefetch_join` can be used to support datasets with many sources.

  Args:
    thread_to_features: List of nest structure of tensors for each thread.
    feed_list: (Optional.) A list of `feed_dict` keys. See
      @{tf.Session.run} for details of the allowable feed key types.
    feed_generator: (Optional.) A generator function lambda sess: iterator
      that yields a list of `feed_dict` values.
    capacity: (Optional.) Max number of samples to keep in the buffer.
    num_clients: (Optional.) Number of clients of prefetched sample. 1 by
      default.
    timeout_millis: (Optional.) Max milliseconds put op can take, 5 min by
      default.
    closed_exception_types: (Optional.) Exception types indicating that the
      prefetching is normally finished. Defaults to
      `(tf.errors.OutOfRangeError, StopIteration)`.
    ignored_exception_types: (Optional.) Exception types indicating that the
      prefetching can continue. Defaults to `()`.
    name: (Optional.) Name of prefetching operations.

  Returns:
    Prefetched sample.
  """
  if len(thread_to_features) < 1:
    raise ValueError('thread_to_features must has at least one element')

  if name is None:
    name = ops.get_default_graph().unique_name(PREFETCH)
  with ops.name_scope(name):
    local_device = control_flow_ops.no_op().device
    with ops.device(local_device):
      cancel_fetching = gen_tensor_buffer_ops.tensor_buffer_cancel(
          shared_name=name,
          shared_capacity=capacity)
      resume_fetching = gen_tensor_buffer_ops.tensor_buffer_cancel(
          is_cancelled=False,
          shared_name=name,
          shared_capacity=capacity)
      close_fetching = gen_tensor_buffer_ops.tensor_buffer_close(
          shared_name=name,
          shared_capacity=capacity)

  thread_to_tensor_dtypes = []
  thread_to_tensor_shapes = []
  thread_to_tensor_or_sparse_tensor_or_nones = []
  thread_to_tensor_or_nones = []
  thread_to_fetch_tensors = []
  for features in thread_to_features:
    tensor_or_sparse_tensor_or_nones = nest.flatten(features)
    thread_to_tensor_or_sparse_tensor_or_nones.append(
        tensor_or_sparse_tensor_or_nones)

    tensor_or_nones = []
    for t in tensor_or_sparse_tensor_or_nones:
      if hasattr(t, 'dense_shape'):
        tensor_or_nones.extend([t.values, t.indices, t.dense_shape])
      else:
        tensor_or_nones.append(t)
    thread_to_tensor_or_nones.append(tensor_or_nones)

    tensor_indices = []
    tensors = []
    for i, v in enumerate(tensor_or_nones):
      if v is not None:
        tensor_indices.append(i)
        tensors.append(v)
    tensor_dtypes = []
    tensor_shapes = []
    for v in tensors:
      tensor_dtypes.append(v.dtype)
      tensor_shapes.append(v.shape if hasattr(v, 'shape') else None)
    thread_to_tensor_dtypes.append(tensor_dtypes)
    thread_to_tensor_shapes.append(tensor_shapes)

    with ops.name_scope(name):
      with ops.device(local_device):
        fetch_tensors = gen_tensor_buffer_ops.tensor_buffer_put(
            tensors,
            timeout_millis=timeout_millis,
            shared_name=name,
            shared_capacity=capacity)
    thread_to_fetch_tensors.append(fetch_tensors)

  with ops.name_scope(name):
    with ops.device(local_device):
      next_tensors = gen_tensor_buffer_ops.tensor_buffer_take(
          dtypes=thread_to_tensor_dtypes[0],
          shared_name=name,
          shared_capacity=capacity,
          shared_threads=num_clients)
      if not isinstance(next_tensors, (tuple, list)):
        next_tensors = [next_tensors]
      next_tensors = [array_ops.identity(t) for t in next_tensors]
    for i, t in enumerate(next_tensors):
      t.set_shape(thread_to_tensor_shapes[0][i])
    next_tensor_or_nones = [None for _ in thread_to_tensor_or_nones[0]]
    for i, v in enumerate(next_tensors):
      next_tensor_or_nones[tensor_indices[i]] = v
    next_tensor_or_nones = collections.deque(next_tensor_or_nones)
    next_tensor_or_sparse_tensor_or_nones = []
    for t in thread_to_tensor_or_sparse_tensor_or_nones[0]:
      if hasattr(t, 'dense_shape'):
        sparse_values = next_tensor_or_nones.popleft()
        sparse_indices = next_tensor_or_nones.popleft()
        sparse_dense_shape = next_tensor_or_nones.popleft()
        next_tensor_or_sparse_tensor_or_nones.append(
            sparse_tensor.SparseTensor(
                values=sparse_values,
                indices=sparse_indices,
                dense_shape=sparse_dense_shape))
      else:
        next_tensor_or_sparse_tensor_or_nones.append(
            next_tensor_or_nones.popleft())
    prefetched = nest.pack_sequence_as(
        thread_to_features[0], next_tensor_or_sparse_tensor_or_nones)

  runner = PrefetchRunner(
      fetch_ops=thread_to_fetch_tensors,
      cancel_op=cancel_fetching,
      resume_op=resume_fetching,
      close_op=close_fetching,
      feed_list=feed_list,
      feed_generator=feed_generator,
      closed_exception_types=closed_exception_types,
      ignored_exception_types=ignored_exception_types)
  ops.add_to_collection(PREFETCH, runner)
  return prefetched
