# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for stateless random ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.platform import test


# Note that in theory each test will reset the eager context and may choose to
# hide some devices, so we shouldn't cache this transient info. Tests in this
# file don't make those config changes, so caching is fine. It provides a good
# speed-up.
_cached_device = None


def get_device():
  global _cached_device
  if _cached_device is not None:
    return _cached_device
  # Precedence from high to low
  for device_type in ('XLA_GPU', 'GPU', 'XLA_CPU', 'CPU'):
    devices = config.list_logical_devices(device_type)
    if devices:
      _cached_device = devices[0]
      return _cached_device
  raise ValueError('Cannot find any suitable device. Available devices: %s' %
                   config.list_logical_devices())


BEFORE_EXPIRE = (2020, 10, 24)
AFTER_EXPIRE = (2020, 10, 26)


def invert_philox(key, value):
  """Invert the Philox bijection."""
  key = np.array(key, dtype=np.uint32)
  value = np.array(value, dtype=np.uint32)
  step = np.array([0x9E3779B9, 0xBB67AE85], dtype=np.uint32)
  for n in range(10)[::-1]:
    key0, key1 = key + n * step
    v0 = value[3] * 0x991a7cdb & 0xffffffff
    v2 = value[1] * 0x6d7cae67 & 0xffffffff
    hi0 = v0 * 0xD2511F53 >> 32
    hi1 = v2 * 0xCD9E8D57 >> 32
    v1 = hi1 ^ value[0] ^ key0
    v3 = hi0 ^ value[2] ^ key1
    value = v0, v1, v2, v3
  return np.array(value)


SEEDS = ((7, 17), (11, 5), (2, 3))
SEED_TYPES = [dtypes.int32, dtypes.int64]


def float_cases(shape_dtypes=(None,)):
  cases = (
      # Uniform distribution, with and without range
      ('uniform', stateless.stateless_random_uniform, random_ops.random_uniform,
       {}),
      ('uniform2', stateless.stateless_random_uniform,
       random_ops.random_uniform, dict(minval=2.2, maxval=7.1)),
      # Normal distribution, with and without mean+stddev
      ('normal', stateless.stateless_random_normal, random_ops.random_normal,
       {}),
      ('normal2', stateless.stateless_random_normal, random_ops.random_normal,
       dict(mean=2, stddev=3)),
      # Truncated normal distribution, with and without mean+stddev
      ('trnorm', stateless.stateless_truncated_normal,
       random_ops.truncated_normal, {}),
      ('trnorm2', stateless.stateless_truncated_normal,
       random_ops.truncated_normal, dict(mean=3, stddev=4)),
  )
  # Explicitly passing in params because capturing cell variable from loop is
  # problematic in Python
  def wrap(op, dtype, shape, shape_dtype, kwds, seed):
    device_type = get_device().device_type
    # Some dtypes are not supported on some devices
    if (dtype == dtypes.float16 and device_type in ('XLA_GPU', 'XLA_CPU') or
        dtype == dtypes.bfloat16 and device_type == 'GPU'):
      dtype = dtypes.float32
    shape_ = (constant_op.constant(shape, dtype=shape_dtype)
              if shape_dtype is not None else shape)
    return op(seed=seed, shape=shape_, dtype=dtype, **kwds)

  def _name(a):
    if hasattr(a, 'name'):
      return a.name
    else:
      return a

  for dtype in dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64:
    for shape_dtype in shape_dtypes:
      for shape in (), (3,), (2, 5):
        for name, stateless_op, stateful_op, kwds in cases:
          yield (('%s_%s_%s_%s' %
                  (name, _name(dtype), shape, _name(shape_dtype))).replace(
                      ' ', ''),
                 functools.partial(wrap, stateless_op, dtype, shape,
                                   shape_dtype, kwds),
                 functools.partial(wrap, stateful_op, dtype, shape, shape_dtype,
                                   kwds))


def int_cases(shape_dtypes=(None,), minval_maxval=None):

  def wrap(op, minval, maxval, shape, shape_dtype, dtype, seed):
    shape_ = (constant_op.constant(shape, dtype=shape_dtype)
              if shape_dtype is not None else shape)
    return op(
        seed=seed, shape=shape_, minval=minval, maxval=maxval, dtype=dtype)

  if minval_maxval is None:
    minval_maxval = ((2, 11111),)
  for minval, maxval in minval_maxval:
    for shape_dtype in shape_dtypes:
      for shape in (), (3,), (2, 5):
        for dtype in dtypes.int32, dtypes.int64:
          yield ('uniform_%s_%s' % (minval, maxval),
                 functools.partial(wrap, stateless.stateless_random_uniform,
                                   minval, maxval, shape, shape_dtype, dtype),
                 functools.partial(wrap, random_ops.random_uniform, minval,
                                   maxval, shape, shape_dtype, dtype))


def multinomial_cases():
  num_samples = 10
  def wrap(op, logits, logits_dtype, output_dtype, seed):
    return op(seed=seed,
              logits=constant_op.constant(logits, dtype=logits_dtype),
              num_samples=num_samples, output_dtype=output_dtype)
  for logits_dtype in np.float16, np.float32, np.float64:
    for output_dtype in dtypes.int32, dtypes.int64:
      for logits in ([[0.1, 0.25, 0.5, 0.15]], [[0.5, 0.5], [0.8, 0.2],
                                                [0.25, 0.75]]):
        yield ('multinomial',
               functools.partial(wrap, stateless.stateless_multinomial, logits,
                                 logits_dtype, output_dtype),
               functools.partial(wrap, random_ops.multinomial, logits,
                                 logits_dtype, output_dtype))

class StatelessOpsTest(test.TestCase, parameterized.TestCase):

  def _test_match(self, case, seed):
    # Stateless ops should be the same as stateful ops on the first call
    # after seed scrambling.
    key = 0x3ec8f720, 0x02461e29
    preseed = invert_philox(key, (seed[0], 0, seed[1], 0)).astype(np.uint64)
    preseed = preseed[::2] | preseed[1::2] << 32
    with ops.device(get_device().name):
      _, stateless_op, stateful_op = case
      random_seed.set_random_seed(seed[0])
      stateful = stateful_op(seed=seed[1])
      pure = stateless_op(seed=preseed)
      self.assertAllEqual(stateful, pure)

  def _test_match_stateless_cpu_gpu(self, case, seed):
    # Stateless ops should produce the same result on CPUs and GPUs.
    _, stateless_op, _ = case

    with ops.device('CPU'):
      result_cpu = stateless_op(seed=seed)

    with ops.device(get_device().name):
      result_gpu = stateless_op(seed=seed)
      self.assertAllClose(result_cpu, result_gpu)

  def _test_old_and_new_stateless_match(self, case, seed):
    """Tests that the new stateless ops match the old stateless ones."""
    with ops.device(get_device().name):
      _, stateless_op, _ = case
      with compat.forward_compatibility_horizon(*BEFORE_EXPIRE):
        old = stateless_op(seed=seed)
      with compat.forward_compatibility_horizon(*AFTER_EXPIRE):
        new = stateless_op(seed=seed)
      self.assertAllClose(old, new)

  def _test_determinism(self, case, seed_type):
    # Stateless values should be equal iff the seeds are equal (roughly)
    seeds = [(x, y) for x in range(5) for y in range(5)] * 3  # pylint: disable=g-complex-comprehension
    with self.test_session(), ops.device(get_device().name):
      _, stateless_op, _ = case
      if context.executing_eagerly():
        values = [
            (seed, stateless_op(seed=constant_op.constant(seed, seed_type)))
            for seed in seeds]
      else:
        # Have this branch because the above branch is too slow in graph
        # mode
        seed_t = array_ops.placeholder(seed_type, shape=[2])
        pure = stateless_op(seed=seed_t)
        values = [
            (seed, pure.eval(feed_dict={seed_t: seed})) for seed in seeds
        ]
      for s0, v0 in values:
        for s1, v1 in values:
          if dtypes.as_dtype(v0.dtype) != dtypes.bfloat16:
            self.assertEqual(s0 == s1, np.all(v0 == v1))
          elif s0 == s1:
            # Skip the s0 != s1 case because v0 and v1 can be either equal or
            # unequal in that case due to bfloat16's low precision
            self.assertAllEqual(v0, v1)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(float_cases()))
  @test_util.run_deprecated_v1
  def testMatchFloat(self, case, seed):
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Skip on XLA because XLA kernels do not support int64 '
                    'seeds needed by this test.')
    self._test_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(int_cases()))
  @test_util.run_deprecated_v1
  def testMatchInt(self, case, seed):
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Skip on XLA because XLA kernels do not support int64 '
                    'seeds needed by this test.')
    self._test_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], case_id, seed_id), case, seed)  # pylint: disable=g-complex-comprehension
      for seed_id, seed in enumerate(SEEDS)
      for case_id, case in enumerate(multinomial_cases()))
  @test_util.run_deprecated_v1
  def testMatchMultinomial(self, case, seed):
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking XLA kernel')
    self._test_match(case, seed)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], seed_type.name, case_id), case, seed_type)  # pylint: disable=g-complex-comprehension
      for seed_type in SEED_TYPES
      for case_id, case in enumerate(
          float_cases(shape_dtypes=(dtypes.int32, dtypes.int64))))
  @test_util.run_deprecated_v1
  def testDeterminismFloat(self, case, seed_type):
    if seed_type == dtypes.int64 and get_device().device_type in ('XLA_GPU',
                                                                  'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest(
          'Skip on XLA because XLA kernels do not support int64 seeds.')
    self._test_determinism(case, seed_type)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], seed_type.name, case_id), case, seed_type)  # pylint: disable=g-complex-comprehension
      for seed_type in SEED_TYPES
      for case_id, case in enumerate(
          int_cases(shape_dtypes=(dtypes.int32, dtypes.int64))))
  @test_util.run_deprecated_v1
  def testDeterminismInt(self, case, seed_type):
    if seed_type == dtypes.int64 and get_device().device_type in ('XLA_GPU',
                                                                  'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest(
          'Skip on XLA because XLA kernels do not support int64 seeds.')
    self._test_determinism(case, seed_type)

  @parameterized.named_parameters(
      ('_%s_%s_%s' % (case[0], seed_type.name, case_id), case, seed_type)  # pylint: disable=g-complex-comprehension
      for seed_type in SEED_TYPES
      for case_id, case in enumerate(multinomial_cases()))
  @test_util.run_deprecated_v1
  def testDeterminismMultinomial(self, case, seed_type):
    if get_device().device_type in ('XLA_GPU', 'XLA_CPU'):
      # This test was passing before because soft placement silently picked the
      # CPU kernels.
      self.skipTest('Lacking XLA kernel')
    self._test_determinism(case, seed_type)

  def assertDTypeEqual(self, a, b):
    self.assertEqual(dtypes.as_dtype(a), dtypes.as_dtype(b))

  def assertNoEqualPair(self, ls):
    for i in range(len(ls)):
      for j in range(i + 1, len(ls)):
        self.assertFalse(math_ops.reduce_all(ls[i] == ls[j]))

  @parameterized.parameters(['int32', 'int64'])
  @test_util.run_v2_only
  def testSplit(self, dtype):
    """Test for `split`."""
    seed = constant_op.constant([1, 2], dtype=dtype)
    new_seed = stateless.split(seed, 3)
    self.assertEqual(new_seed.shape, [3, 2])
    self.assertDTypeEqual(new_seed.dtype, dtype)
    self.assertNoEqualPair([seed] + array_ops.unstack(new_seed))

  @parameterized.parameters(['int32', 'int64'])
  @test_util.run_v2_only
  def testFoldIn(self, dtype):
    """Test for `fold_in`."""
    orig_seed = constant_op.constant([1, 2], dtype='int32')
    seed = stateless.fold_in(orig_seed, constant_op.constant(3, dtype=dtype))
    new_seeds = []
    new_seeds.append(seed)
    seed = stateless.fold_in(seed, constant_op.constant(4, dtype=dtype))
    new_seeds.append(seed)
    for s in new_seeds:
      self.assertEqual(s.shape, [2])
      self.assertDTypeEqual(s.dtype, dtype)
    self.assertNoEqualPair([math_ops.cast(orig_seed, dtype)] + new_seeds)

  @test_util.run_v2_only
  def testErrors(self):
    """Tests that proper errors are raised.
    """
    shape = [2, 3]
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        'minval must be a scalar; got a tensor of shape '):
      @def_function.function
      def f():
        stateless.stateless_random_uniform(
            shape=shape, seed=[1, 2], minval=array_ops.zeros(shape, 'int32'),
            maxval=100, dtype='int32')
      f()
    with self.assertRaisesWithPredicateMatch(
        ValueError,
        'maxval must be a scalar; got a tensor of shape '):
      @def_function.function
      def f2():
        stateless.stateless_random_uniform(
            shape=shape, seed=[1, 2], minval=0,
            maxval=array_ops.ones(shape, 'int32') * 100,
            dtype='int32')
      f2()


if __name__ == '__main__':
  config.set_soft_device_placement(False)
  test.main()
