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

import itertools
import threading
import weakref

from six.moves import xrange

from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.core.protobuf import config_pb2

class PrefetchRunner(object): # pylint: disable=useless-object-inheritance
  """Prefetch tensors by repeating running given ops.

  The `PrefetchRunner`, combined with the `Coordinator` provides a way to
  compute tensors asynchronously using multiple threads.
  """

  class Hook(session_run_hook.SessionRunHook):
    """SessionRunHook that starts prefetch runners after session creation."""
    def __init__(self, collection, daemon=True, start=True):
      """Build PrefetchRunner.Hook.

      Args:
        collection: Name of the runner collection.
        daemon: (Optional.) Whether the threads should be marked as `daemons`,
          meaning they don't block program exit.
        start: (Optional.) If `False` threads would not be started.
      """
      super(PrefetchRunner.Hook, self).__init__()
      self._collection = collection
      self._daemon = daemon
      self._start = start

    def after_create_session(self, session, coord):
      self.create_threads(sess=session, coord=coord)

    def create_threads(self, sess=None, coord=None):
      """Create threads for runners in specific collection.

      It starts threads for all runners collected in the graph. It returns
      the list of all threads.

      Args:
        sess: `Session` used to run the stage ops. Defaults to the
          default session.
        coord: (Optional.) `Coordinator` for coordinating the started threads.

      Raises:
        ValueError: if `sess` is None and there isn't any default session.
        TypeError: if `sess` is not a `tf.Session` object.

      Returns:
        A list of threads.
      """
      if sess is None:
        sess = ops.get_default_session()
        if not sess:
          raise ValueError("Cannot start threads: No default session is "
                           "registered. Use `with sess.as_default()` or use "
                           "explicit session in create_threads")

      if not isinstance(sess, session_lib.SessionInterface):
        if sess.__class__.__name__ in [
            "MonitoredSession", "SingularMonitoredSession"]:
          return []
        raise TypeError("sess must be a `tf.Session` object. "
                        "Given class: {}".format(sess.__class__))

      with sess.graph.as_default():
        threads = []
        for runner in ops.get_collection(self._collection):
          threads.extend(runner.create_threads(
              sess, coord=coord, daemon=self._daemon, start=self._start))
      return threads

  def __init__(
      self,
      fetch_ops,
      cancel_op,
      resume_op,
      close_op,
      feed_list=None,
      feed_generator=None,
      closed_exception_types=None,
      ignored_exception_types=None,
      use_stage_subgraph_thread_pool=False,
      stage_subgraph_thread_pool_id=0):
    """Create a PrefetchRunner.

    When you later call the `create_threads()` method, the `PrefetchRunner` will
    create threads for `fetch_ops`. Each thread will prefetch
    in parallel.

    Args:
      fetch_ops: Ops that repeats by this runner for each thread.
      cancel_op: Op that stops fetch_ops on stop of this runner.
      resume_op: Op that restarts fetch_ops on start of this runner.
      close_op: Op that closes the data buffer.
      feed_list: (Optional.) A list of `feed_dict` keys. See
        @{tf.Session.run} for details of the allowable feed key types.
      feed_generator: (Optional.) A generator function lambda sess: iterator
        that yields a list of `feed_dict` values.
      closed_exception_types: (Optional.) Exception types indicating that the
        prefetching is normally finished. Defaults to
        `(tf.errors.OutOfRangeError, StopIteration)`.
      ignored_exception_types: (Optional.) Exception types indicating that the
        prefetching can continue. Defaults to `()`.
      use_stage_subgraph_thread_pool: (Optional.) Use stage subgraph thread pool
        to run stage graph or not.
      stage_subgraph_thread_pool_id: (Optional.) Specifies the stage subgraph
        thread pool to use when enable use_stage_subgraph_thread_pool. 0 by default.
    """
    try:
      executing_eagerly = context.executing_eagerly()
    except: # pylint: disable=bare-except
      executing_eagerly = context.in_eager_mode()
    else:
      executing_eagerly = False
    if not executing_eagerly:
      self._name = ops.get_default_graph().unique_name(self.__class__.__name__)
    else:
      self._name = context.context().scope_name
    self._fetch_ops = fetch_ops
    self._cancel_op = cancel_op
    self._resume_op = resume_op
    self._close_op = close_op
    if (feed_list is None) != (feed_generator is None):
      raise ValueError("feed_list and feed_generator must both exits")
    self._feed_list = list(feed_list) if feed_list else None
    self._feed_generator = feed_generator
    if not closed_exception_types:
      self._closed_exception_types = (errors.OutOfRangeError, StopIteration)
    else:
      self._closed_exception_types = tuple(closed_exception_types)
    if not ignored_exception_types:
      self._ignored_exception_types = ()
    else:
      self._ignored_exception_types = tuple(ignored_exception_types)
    self._lock = threading.Lock()
    self._runs_per_session = weakref.WeakKeyDictionary()
    self._exceptions_raised = []
    self._use_stage_subgraph_thread_pool = use_stage_subgraph_thread_pool
    self._stage_subgraph_thread_pool_id = stage_subgraph_thread_pool_id

  @property
  def name(self):
    """Name of this runner."""
    return self._name

  @property
  def feed_list(self):
    """List of feeds used in prefetch ops."""
    return self._feed_list

  @property
  def num_threads(self):
    """The number of running threads."""
    return len(self._fetch_ops)

  @property
  def closed_exception_types(self):
    """Exception types indicating that prefetching is normally finished."""
    return self._closed_exception_types

  @property
  def exceptions_raised(self):
    """Exceptions raised but not handled by the `PrefetchRunner` threads.

    Exceptions raised in `PrefetchRunner` threads are handled in one of two ways
    depending on whether or not a `Coordinator` was passed to
    `create_threads()`:

    * With a `Coordinator`, exceptions are reported to the coordinator and
      forgotten by the `PrefetchRunner`.
    * Without a `Coordinator`, exceptions are captured by the `PrefetchRunner`
      and made available in this `exceptions_raised` property.

    Returns:
      A list of Python `Exception` objects.  The list is empty if no exception
      was captured.  (No exceptions are captured when using a Coordinator.)
    """
    return self._exceptions_raised

  # pylint: disable=broad-except
  def run(self, sess, coord, index):
    """Run prefetching in thread.

    Args:
      sess: A `Session`.
      coord: A `Coordinator` object for reporting errors and checking stop
        conditions.
      index: Index of current thread.
    """
    decremented = False
    try:
      sess.run(self._resume_op)
      run_fetch = sess.make_callable(
          self._fetch_ops[index], self._feed_list, True)
      close = sess.make_callable(self._close_op)
      feed_list = self._feed_list if self._feed_list else []
      if self._feed_generator:
        feed_iterator = self._feed_generator(sess)
      else:
        feed_iterator = itertools.repeat([])
      run_options = config_pb2.RunOptions()
      run_options.use_stage_subgraph_thread_pool = self._use_stage_subgraph_thread_pool
      run_options.stage_subgraph_thread_pool_id = self._stage_subgraph_thread_pool_id
      while True:
        try:
          # Use `next` instead of `for .. in` to reraise exception in generator.
          feed = next(feed_iterator)
          if coord and coord.should_stop():
            break
          if not isinstance(feed, (list, tuple)):
            raise ValueError(
                'feed_generator must generate a tuple, not {} ({})'.format(
                    feed, type(feed).__name__))
          if len(feed) != len(feed_list):
            raise ValueError(
                'feed_generator must generate a tuple of {} items, not {} '
                '({} items)'.format(
                    len(feed_list), feed, len(feed)))
          run_fetch(*feed, options=run_options)
        except errors.CancelledError:
          logging.info("Prefetching was cancelled.")
          return
        except self._closed_exception_types as e:  # pylint: disable=catching-non-exception
          logging.info("Prefetching was closed.")
          with self._lock:
            self._runs_per_session[sess] -= 1
            decremented = True
            if self._runs_per_session[sess] == 0:
              try:
                close()
              except Exception:
                pass
            return
        except self._ignored_exception_types as e:  # pylint: disable=catching-non-exception
          logging.warning(
              "Corrupted inputs were ignored in prefetching:\n\n%s", e)
          continue
    except Exception as e:
      if coord:
        coord.request_stop(e)
        if not isinstance(e, errors.CancelledError) and \
           not isinstance(e, self._closed_exception_types) and \
           not isinstance(e, self._ignored_exception_types):
          logging.error(
              "Prefetching was cancelled unexpectedly:\n\n%s", e)
          raise
      else:
        with self._lock:
          self._exceptions_raised.append(e)
        raise
    finally:
      if not decremented:
        with self._lock:
          self._runs_per_session[sess] -= 1

  def cancel_on_stop(self, sess, coord):
    """Clean up resources on stop.

    Args:
      sess: A `Session`.
      coord: A `Coordinator` object for reporting errors and checking stop
        conditions.
    """
    coord.wait_for_stop()
    try:
      cancel = sess.make_callable(self._cancel_op)
      cancel()
    except Exception:
      pass
  # pylint: enable=broad-except

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    """Create threads to prefetch for the given session.

    This method requires a session in which the graph was launched. It creates
    a list of threads, optionally starting them.

    The `coord` argument is an optional coordinator that the threads will use
    to terminate together and report exceptions.  If a coordinator is given,
    this method starts an additional thread to cancel when the coordinator
    requests a stop.

    If previously created threads for the given session are still running, no
    new threads will be created.

    Args:
      sess: A `Session`.
      coord: (Optional.) `Coordinator` object for reporting errors and checking
        stop conditions.
      daemon: (Optional.) Boolean. If `True` make the threads daemon threads.
      start: (Optional.) Boolean. If `True` starts the threads.  If `False` the
        caller must call the `start()` method of the returned threads.

    Returns:
      A list of threads.
    """
    with self._lock:
      try:
        if self._runs_per_session[sess] > 0:
          # Already started: no new threads to return.
          return []
      except KeyError:
        pass
      self._runs_per_session[sess] = self.num_threads
      self._exceptions_raised = []

    ret_threads = []
    for i in xrange(self.num_threads):
      ret_threads.append(threading.Thread(
          target=self.run,
          args=(sess, coord, i),
          name="PrefetchThread-%s-%s" % (self.name, i)))
    if coord:
      name = "CancelOnStopThread-%s" % self.name
      ret_threads.append(threading.Thread(
          target=self.cancel_on_stop,
          args=(sess, coord),
          name=name))
    for t in ret_threads:
      if coord:
        coord.register_thread(t)
      if daemon:
        t.daemon = True
      if start:
        t.start()
    return ret_threads
