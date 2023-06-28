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
"""Prefetch runner for prefetching ops.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.framework import ops
from tensorflow.python.training import session_run_hook
from tensorflow.python import pywrap_tensorflow as prefetch_runner

class PrefetchRunnerHook(session_run_hook.SessionRunHook):
  """
  PrefetchRunnerHook that starts prefetch runners after session creation and
  stops prefetch runners before session close.
  """
  def __init__(self):
    """Build PrefetchRunnerHook."""
    super(PrefetchRunnerHook, self).__init__()

  def cancel_on_stop(self, session, coord, graph_key):
    coord.wait_for_stop()
    prefetch_runner.TF_StopPrefetchRunners(graph_key, session.c_session)

  def after_create_session(self, session, coord):
    session._extend_graph()
    graph_key = ops.get_default_graph()._graph_key
    prefetch_runner.TF_StartPrefetchRunners(graph_key, session.c_session)
    self._stop_thread = threading.Thread(target=self.cancel_on_stop,
                                         args=(session, coord, graph_key),
                                         name="prefetch_runner_cancel_on_stop")
    coord.register_thread(self._stop_thread)
    self._stop_thread.daemon=True
    self._stop_thread.start()
