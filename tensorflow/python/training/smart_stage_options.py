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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export

PREFETCH = "prefetch"

@tf_export(v1=["SmartStageOptions"])
def SmartStageOptions(
    capacity=1,
    num_threads=1,
    num_clients=1,
    timeout_millis=300000,
    closed_exception_types=(errors.OUT_OF_RANGE,),
    ignored_exception_types=(),
    use_stage_subgraph_thread_pool=False,
    stage_subgraph_thread_pool_id=0,
    stage_subgraph_stream_id=0,
    graph=None,
    name=None):
  """Generate SmartStageOptions.

  Args:
    capacity: (Optional.) Max number of samples to keep in the buffer.
    num_threads: (Optional.) Number of threads for prefetching. 1 by
      default.
    num_clients: (Optional.) Number of clients of prefetched sample. 1 by
      default.
    timeout_millis: (Optional.) Max milliseconds put op can take, 5 min by
      default.
    closed_exception_types: (Optional.) Exception types indicating that the
      prefetching is normally finished. Defaults to
      `(errors.OUT_OF_RANGE,)`.
    ignored_exception_types: (Optional.) Exception types indicating that the
      prefetching can continue. Defaults to `()`.
    use_stage_subgraph_thread_pool: (Optional.) Use stage subgraph thread pool
      to run stage graph or not.
    stage_subgraph_thread_pool_id: (Optional.) Specifies the stage subgraph
      thread pool to use when enable use_stage_subgraph_thread_pool. 0 by default.
    stage_subgraph_stream_id: (Optional.) Specifies which stream to use for the
      Stage subgraph. The default value is 0.
    graph: (Optional.) Specify the graph for SmartStage, which is the graph
      passed to the Session.
    name: (Optional.) Name of prefetching operations.

  Returns:
    SmartStageOptions.
  """
  options = config_pb2.SmartStageOptions()
  if capacity < 1:
    raise ValueError('capacity must >= 1')
  options.capacity = capacity

  if num_threads < 1:
    raise ValueError('num_threads must >= 1')
  options.num_threads = num_threads

  if num_clients < 1:
    raise ValueError('num_clients must >= 1')
  options.num_clients = num_clients

  if timeout_millis <= 0:
    raise ValueError('timeout_millis must > 0')
  options.timeout_millis = timeout_millis

  for err_code in closed_exception_types:
    options.runner_options.closed_exceptions.append(err_code)

  for err_code in ignored_exception_types:
    options.runner_options.ignored_exceptions.append(err_code)

  options.runner_options.run_options.use_stage_subgraph_thread_pool = \
    use_stage_subgraph_thread_pool

  if stage_subgraph_thread_pool_id < 0:
    raise ValueError('stage_subgraph_thread_pool_id must >= 0')
  options.runner_options.run_options.stage_subgraph_thread_pool_id = \
    stage_subgraph_thread_pool_id

  if stage_subgraph_stream_id < 0:
    raise ValueError('stage_subgraph_stream_id >= 0')
  options.stage_subgraph_stream_id = stage_subgraph_stream_id

  if graph is None:
    graph = ops.get_default_graph()
  options.graph_key = graph._graph_key

  if name is None:
    name = ops.get_default_graph().unique_name(PREFETCH)
  options.name = name

  return options
