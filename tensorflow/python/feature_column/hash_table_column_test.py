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
"""Tests for feature_column."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import collections
import copy

import numpy as np
import os
import tempfile

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.feature_column import dense_features as df
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import hash_table_column as hc
from tensorflow.python.feature_column import serialization
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.hash_table import hash_table, hash_filter
from tensorflow.python.ops.hash_table.embedding import ReadOnlyHook
from tensorflow.python.platform import test
from tensorflow.python.training import rmsprop
from tensorflow.python.training import adagrad
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
from tensorflow.python.training.training_util import get_or_create_global_step
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.util import tf_inspect


def _initialized_session(config=None):
  sess = session.Session(config=config)
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess

def LocalTempDir():
  """Return a temporary directory for tests to use."""
  first_frame = tf_inspect.stack()[-1][0]
  temp_dir = os.path.join(tempfile.gettempdir(),
                          os.path.basename(tf_inspect.getfile(first_frame)))
  temp_dir = tempfile.mkdtemp(prefix=temp_dir.rstrip('.py'))
  def delete_temp_dir(dirname=temp_dir):
    try:
      file_io.delete_recursively(dirname)
    except errors.OpError as e:
      logging.error('Error removing %s: %s', dirname, e)
  atexit.register(delete_temp_dir)
  return temp_dir

class HashTableColumnTest(test.TestCase):
  def test_defaults(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    dimension = 2
    column = hc.hash_table_column(categorical_column, dimension,
                                  dtypes.float32,
                                  init_ops.ones_initializer())
    self.assertIs(categorical_column, column.categorical_column)
    self.assertEqual(dimension, column.dimension)
    self.assertEqual('mean', column.combiner)
    self.assertTrue(column.trainable)
    self.assertEqual('aaa_hash_table', column.name)
    self.assertEqual('aaa', column.table_name)
    self.assertIsNone(column.coalesced_scope)
    self.assertEqual({'aaa': parsing_ops.VarLenFeature(dtypes.int64)},
                     column.parse_example_spec)

  def test_all_constructor_args(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    dimension = 2
    column = hc.hash_table_column(
        categorical_column, dimension, dtype=dtypes.float32,
        combiner='my_combiner', initializer=lambda : 'my_initializer',
        partitioner=lambda : 'my_partitioner', trainable=False)
    self.assertIs(categorical_column, column.categorical_column)
    self.assertEqual(dimension, column.dimension)
    self.assertEqual(dtypes.float32, column.dtype)
    self.assertEqual('my_combiner', column.combiner)
    self.assertFalse(column.trainable)
    self.assertEqual('aaa_hash_table', column.name)
    self.assertEqual('aaa', column.table_name)
    self.assertIsNone(column.coalesced_scope)
    self.assertEqual((dimension,), column.variable_shape)
    self.assertEqual({'aaa': parsing_ops.VarLenFeature(dtypes.int64)},
                     column.parse_example_spec)

  def test_deep_copy(self):
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    dimension = 2
    original = hc.hash_table_column(
        categorical_column, dimension, dtype=dtypes.float32,
        combiner='my_combiner', initializer=lambda : 'my_initializer',
        partitioner=lambda : 'my_partitioner', trainable=False)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.categorical_column.name)
      self.assertEqual(3, column.categorical_column._num_buckets)
      self.assertEqual({'aaa': parsing_ops.VarLenFeature(dtypes.int64)},
                       column.categorical_column.parse_example_spec)

      self.assertEqual(dimension, column.dimension)
      self.assertEqual(dtypes.float32, column.dtype)
      self.assertEqual('my_combiner', column.combiner)
      self.assertFalse(column.trainable)
      self.assertEqual('aaa_hash_table', column.name)
      self.assertEqual('aaa', column.table_name)
      self.assertIsNone(column.coalesced_scope)
      self.assertEqual((dimension,), column.variable_shape)
      self.assertEqual({'aaa': parsing_ops.VarLenFeature(dtypes.int64)},
                       column.parse_example_spec)

  def test_parse_example(self):
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    ht = hc.hash_table_column(a, 2)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([ht]))
    self.assertIn('aaa', features)
    with self.cached_session():
      _assert_sparse_tensor_value(
          self,
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [0, 1]],
              values=np.array([b'omar', b'stringer'], dtype=np.object_),
              dense_shape=[1, 2]),
          features['aaa'].eval())

  def test_transform_feature(self):
    a = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    ht = hc.hash_table_column(a, dimension=2)
    features = {
        'aaa': sparse_tensor.SparseTensor(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((0, 1, 0)),
            dense_shape=(2, 2))
    }
    outputs = fc._transform_features_v2(features, (a, ht), None)
    output_a = outputs[a]
    output_ht = outputs[ht]
    with _initialized_session():
      _assert_sparse_tensor_value(
          self, output_a.eval(), output_ht.eval())

  def test_shared_hash_table(self):
    # Build columns.
    aaa = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    bbb = fc.categorical_column_with_identity(
        key='bbb', num_buckets=5)
    ccc = fc.categorical_column_with_identity(
        key='ccc', num_buckets=10)
    ha = hc.shared_hash_table_column(
        aaa, dimension=2,
        shared_name='shared1',
        initializer=init_ops.ones_initializer(),
        combiner='sum')
    hb = hc.shared_hash_table_column(
        bbb, dimension=2,
        shared_name='shared1',
        initializer=init_ops.ones_initializer(),
        combiner='sum')
    shared2 = hc.shared_hash_table_columns(
        [bbb, ccc], dimension=3,
        shared_name='shared2',initializer=init_ops.ones_initializer(),
        combiner='sum')

    state_manager = _TestStateManager()
    ha.create_state(state_manager)
    hb.create_state(state_manager)
    shared2[0].create_state(state_manager)
    shared2[1].create_state(state_manager)
    inputs = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((2, 3, 4)),
            dense_shape=(2, 2)),
        'ccc': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (0, 1), (1, 0)),
            values=np.array((7, 8, 9)),
            dense_shape=(3, 3))
    }
    e1 = ha.get_dense_tensor(fc.FeatureTransformationCache(inputs), state_manager)
    e2 = hb.get_dense_tensor(fc.FeatureTransformationCache(inputs), state_manager)
    e3 = shared2[0].get_dense_tensor(fc.FeatureTransformationCache(inputs), state_manager)
    e4 = shared2[1].get_dense_tensor(fc.FeatureTransformationCache(inputs), state_manager)

    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('shared1/HashTable_0',
                           'shared2/HashTable_0'),
                          tuple([v.name for v in global_vars]))

    expected1 = ((1.,1.),(2.,2.),(0.,0.),(1.,1.))
    expected2 = ((1.,1.),(2.,2.))
    expected3 = ((1.,1.,1.),(2.,2.,2.))
    expected4 = ((2.,2.,2.),(1.,1.,1.),(0.,0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected1, e1.eval())
      self.assertAllEqual(expected2, e2.eval())
      self.assertAllEqual(expected3, e3.eval())
      self.assertAllEqual(expected4, e4.eval())

  def test_column_with_variable_scope(self):
    # Build columns.
    embedding_dimension = 2
    aaa = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    bbb = fc.categorical_column_with_identity(
        key='bbb', num_buckets=5)

    inputs = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((2, 3, 4)),
            dense_shape=(2, 2))
    }
    state_manager = _TestStateManager()
    with variable_scope.variable_scope('v1', reuse=variable_scope.AUTO_REUSE):
      h1 = hc.hash_table_column(
          aaa, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
      h1.create_state(state_manager)
      embedding_1 = h1.get_dense_tensor(
        fc.FeatureTransformationCache(inputs), state_manager)

    with variable_scope.variable_scope('v1', reuse=variable_scope.AUTO_REUSE):
      h2 = hc.hash_table_column(
          bbb, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
      h2.create_state(state_manager)
      embedding_2 = h2.get_dense_tensor(
        fc.FeatureTransformationCache(inputs), state_manager)

    with variable_scope.variable_scope('v1', reuse=variable_scope.AUTO_REUSE):
      h3 = hc.hash_table_column(
          aaa, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
      h3.create_state(state_manager)
      embedding_3 = h2.get_dense_tensor(
        fc.FeatureTransformationCache(inputs), state_manager)

    with variable_scope.variable_scope('v2', reuse=variable_scope.AUTO_REUSE):
      h4 = hc.hash_table_column(
          bbb, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
      h4.create_state(state_manager)
      embedding_4 = h2.get_dense_tensor(
        fc.FeatureTransformationCache(inputs), state_manager)

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('v1/aaa/HashTable_0',
                           'v1/bbb/HashTable_0',
                           'v2/bbb/HashTable_0'),
                          tuple([v.name for v in global_vars]))

  def test_incompatible_shared_hash_table(self):
    # Build columns.
    aaa = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    bbb = fc.categorical_column_with_identity(
        key='aaa', num_buckets=5)
    ha = hc.shared_hash_table_column(
        aaa, dimension=2,
        initializer=init_ops.ones_initializer(),
        combiner='sum', shared_name='shared_embedding')
    hb = hc.shared_hash_table_column(
        bbb, dimension=3,
        initializer=init_ops.ones_initializer(),
        combiner='sum', shared_name='shared_embedding')

    inputs = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 1), (1, 2)),
            values=np.array((2, 3, 4)),
            dense_shape=(2, 2))
    }
    state_manager = _TestStateManager()
    ha.create_state(state_manager)
    hb.create_state(state_manager)
    with _initialized_session():
      with self.assertRaisesRegexp(
          ValueError,
          'Cannot share HashTable with incompatible dimension'):
        embedding_a = ha.get_dense_tensor(
          fc.FeatureTransformationCache(inputs), state_manager)
        embedding_b = hb.get_dense_tensor(
          fc.FeatureTransformationCache(inputs), state_manager)

  def test_get_dense_tensor(self):
    # Inputs.
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=np.array((2, 0, 1, 1)),
        dense_shape=(5, 5))

    # Build columns.
    embedding_dimension = 2
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    ht = hc.hash_table_column(
        categorical_column, dimension=embedding_dimension,
        initializer=init_ops.ones_initializer(),
        combiner='sum')

    state_manager = _TestStateManager()
    ht.create_state(state_manager)
    embedding_lookup = ht.get_dense_tensor(
      fc.FeatureTransformationCache({"aaa": sparse_input}), state_manager)

    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('aaa/HashTable_0',),
                          tuple([v.name for v in global_vars]))

    expected = ((1.,1.),(2.,2.),(0.,0.),(1.,1.),(0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected, embedding_lookup.eval())

  def test_weighted_categorical_column(self):
    # Inputs.
    vocabulary_size = 3
    sparse_ids = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=np.array((2, 0, 1, 1)),
        dense_shape=(5, 5))
    sparse_weights = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=np.array((1, 2, 3, 4), dtype=np.float32),
        dense_shape=(5, 5))

    # Build columns.
    embedding_dimension = 2
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    weighted_column = fc.weighted_categorical_column(
        categorical_column, 'aaa_weight')
    ht = hc.hash_table_column(
        weighted_column, dimension=embedding_dimension,
        initializer=init_ops.ones_initializer(),
        combiner='sum')

    state_manager = _TestStateManager()
    ht.create_state(state_manager)
    embedding_lookup = ht.get_dense_tensor(
      fc.FeatureTransformationCache({"aaa": sparse_ids,
                                     "aaa_weight": sparse_weights}), state_manager)

    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('aaa/HashTable_0',),
                          tuple([v.name for v in global_vars]))

    expected = ((1.,1.),(5.,5.),(0.,0.),(4.,4.),(0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected, embedding_lookup.eval())

  def test_dense_features(self):
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 5))

    embedding_dimension = 2
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    ht = hc.hash_table_column(
        categorical_column, dimension=embedding_dimension,
        initializer=init_ops.ones_initializer(),
        combiner='mean')
    dense_features = df.DenseFeatures(
        feature_columns=(ht,))(
        {'aaa': sparse_input})

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',),
                          tuple([v.name for v in global_vars]))
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',),
                          tuple([v.name for v in trainable_vars]))

    expected = ((1.,1.),(1.,1.),(0.,0.),(1.,1.))
    with _initialized_session():
      self.assertAllEqual(expected, dense_features.eval())
 
  def test_dense_features_with_readonly_hook(self):
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 5))

    embedding_dimension = 2
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    lookup_hook = ReadOnlyHook()
    ht = hc.hash_table_column(
        categorical_column, dimension=embedding_dimension,
        initializer=init_ops.ones_initializer(),
        combiner='mean', embedding_lookup_hooks=lookup_hook)
    dense_features = df.DenseFeatures(
        feature_columns=(ht,))(
        {'aaa': sparse_input})

    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',),
                          tuple([v.name for v in global_vars]))
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',),
                          tuple([v.name for v in trainable_vars]))

    expected = ((0.,0.),(0.,0.),(0.,0.),(0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected, dense_features.eval())

  def test_dense_features_with_global_filter(self):
    vocabulary_size = 1000
    tensors = []
    for i in range(3):
      offset = i*4
      tensors.append(constant_op.constant([offset, offset+1, offset+2, offset+3], dtype=dtypes.int64))
    dataset = dataset_ops.Dataset.from_tensor_slices(tensors).repeat()
    input_ids = dataset.make_one_shot_iterator().get_next()
    sparse_input = sparse_tensor.SparseTensorValue(
            indices=((0,0), (1,0), (2,0), (3,0)),
            values=input_ids,
            dense_shape=(4,1))

    embedding_dimension = 4
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    lookup_hook = hash_filter.GlobalStepFilter(1)
    ht = hc.hash_table_column(
        categorical_column, dimension=embedding_dimension,
        initializer=init_ops.ones_initializer(),
        combiner='mean',
        embedding_lookup_hooks=lookup_hook)
    dense_features = df.DenseFeatures(
        feature_columns=(ht,))(
        {'aaa': sparse_input})
    loss = math_ops.reduce_mean(dense_features, 0)
    opt = adagrad.AdagradOptimizer(0.1)
    global_step = get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)

    filter_hook = hash_filter.HashFilterHook(is_chief=True)
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    snapshot = trainable_vars[0].snapshot
    with MonitoredTrainingSession('', hooks=[filter_hook]) as sess:
      for i in range(10):
        sess.run(train_op)
      hash_filter.filter_once(sess)
      left_ids = sess.run(snapshot)[0]
      self.assertTrue(np.sort(left_ids).tolist() == [0,1,2,3])

  def test_dense_features_with_concat(self):
    sparse_a = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 5))
    sparse_b = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (0, 1), (2, 0), (2, 1)),
        values=(0, 1, 2, 3),
        dense_shape=(4, 3))
    categorical_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=4)

    ha = hc.hash_table_column(
        categorical_a, dimension=2,
        initializer=init_ops.ones_initializer(),
        combiner='mean')

    hb = hc.hash_table_column(
        categorical_b, dimension=3,
        initializer=init_ops.ones_initializer(),
        combiner='sum')
    dense_features = df.DenseFeatures(
        feature_columns=(ha, hb))(
        {'aaa': sparse_a, 'bbb': sparse_b})
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',
                           'dense_features/bbb/bbb/HashTable_0'),
                          tuple([v.name for v in global_vars]))
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',
                           'dense_features/bbb/bbb/HashTable_0'),
                          tuple([v.name for v in trainable_vars]))
    expected = (
        (1.,1.,2.,2.,2.),
        (1.,1.,0.,0.,0.),
        (0.,0.,2.,2.,2.),
        (1.,1.,0.,0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected, dense_features.eval())

  def test_reuse_categorical_column(self):
    sparse_input = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(5, 5))
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)

    ha = hc.hash_table_column(
        categorical_column, dimension=2,
        initializer=init_ops.ones_initializer(),
        combiner='mean')

    hb = hc.shared_hash_table_column(
        categorical_column, dimension=3,
        initializer=init_ops.ones_initializer(),
        combiner='sum', shared_name='bbb')
    dense_features = df.DenseFeatures(
        feature_columns=(ha, hb))(
        {'aaa': sparse_input})
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',
                           'dense_features/bbb/bbb/HashTable_0'),
                          tuple([v.name for v in global_vars]))
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',
                           'dense_features/bbb/bbb/HashTable_0'),
                          tuple([v.name for v in trainable_vars]))
    expected = (
        (1.,1.,1.,1.,1.),
        (1.,1.,2.,2.,2.),
        (0.,0.,0.,0.,0.),
        (1.,1.,1.,1.,1.),
        (0.,0.,0.,0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected, dense_features.eval())

  def test_reuse_and_shared_column(self):
    sparse_a = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 5))
    sparse_b = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (0, 1), (2, 0), (2, 1)),
        values=(0, 1, 2, 3),
        dense_shape=(4, 3))

    categorical_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=4)

    h1 = hc.hash_table_column(
        categorical_a, dimension=2,
        initializer=init_ops.ones_initializer(),
        combiner='mean')

    h2 = hc.shared_hash_table_column(
        categorical_a, dimension=3,
        initializer=init_ops.ones_initializer(),
        combiner='sum', shared_name='bbb')

    h3 = hc.shared_hash_table_column(
        categorical_b, dimension=3,
        initializer=init_ops.ones_initializer(),
        combiner='sum', shared_name='bbb')
    dense_features = df.DenseFeatures(
        feature_columns=(h1, h2, h3))(
        {'aaa': sparse_a, 'bbb': sparse_b})
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',
                           'dense_features/bbb/bbb/HashTable_0'),
                          tuple([v.name for v in global_vars]))
    trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertItemsEqual(('dense_features/aaa/aaa/HashTable_0',
                           'dense_features/bbb/bbb/HashTable_0'),
                          tuple([v.name for v in trainable_vars]))
    expected = (
        (1.,1.,1.,1.,1.,2.,2.,2.),
        (1.,1.,2.,2.,2.,0.,0.,0.),
        (0.,0.,0.,0.,0.,2.,2.,2.),
        (1.,1.,1.,1.,1.,0.,0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected, dense_features.eval())

  def test_coalesce_encode_ids(self):
    embedding_dimension = 2
    categorical_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=5)
    weighted_b = fc.weighted_categorical_column(
        categorical_b, 'bbb_weight')

    ha = hc.hash_table_column(
        categorical_a, dimension=embedding_dimension,
        initializer=init_ops.ones_initializer(),
        combiner='sum')
    hb = hc.hash_table_column(
        weighted_b, dimension=embedding_dimension,
        initializer=init_ops.ones_initializer(),
        combiner='sum')
    coalesced_column = hc.CoalescedHashTableColumn([ha, hb])
    inputs = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((1, 0), (2, 0), (3, 0), (4, 0)),
            values=np.array((1, 2, 3, 4)),
            dense_shape=(5, 2)),
        'bbb_weight': sparse_tensor.SparseTensorValue(
            indices=((1, 0), (2, 0), (3, 0), (4, 0)),
            values=np.array((4, 3, 2, 1), dtype=np.float32),
            dense_shape=(5, 2)),
    }
    lookup_input_list = coalesced_column.make_sparse_inputs(
        fc.FeatureTransformationCache(inputs), None)
    merged_ids, merged_weights, size_list, origin_shape_list = lookup_input_list[0]
    b = (1 << hc.MASK_REMAIN_LENGTH)
    indices= (
        (0,0),(1,0),(1,1),(3,0),
        (5,0),(6,0),(7,0),(8,0))
    ids  = np.array((2,0,1,1,b+1,b+2,b+3,b+4), dtype=np.int64)
    expected_ids = sparse_tensor.SparseTensor(
        indices=indices, values=ids, dense_shape=(9, 5))
    expected_weights = sparse_tensor.SparseTensor(
        indices=indices, values=np.array((1,1,1,1,4,3,2,1), dtype=np.float32),
        dense_shape=(9, 5))
    with _initialized_session():
      _assert_sparse_tensor_value(self, expected_ids.eval(), merged_ids.eval())
      _assert_sparse_tensor_value(
          self, expected_weights.eval(), merged_weights.eval())
      self.assertEqual(4, size_list[0].eval())
      self.assertEqual(5, size_list[1].eval())

  def test_simple_coalesce(self):
    # Inputs.
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (3, 0)),
        values=np.array((2, 0, 1, 1)),
        dense_shape=(4, 5))

    # Build columns.
    embedding_dimension = 2
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    with hc.coalesced_hash_table_scope():
      ht = hc.hash_table_column(
          categorical_column, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(), combiner='sum')
    state_manager = _TestStateManager()
    ht.create_state(state_manager)
    embedding_lookup = ht.get_dense_tensor(
        fc.FeatureTransformationCache({
            'aaa': sparse_input
        }), state_manager)

    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('CoalescedHashTable/HashTable_0',),
                          tuple([v.name for v in global_vars]))

    expected = ((1.,1.),(2.,2.),(0.,0.),(1.,1.))
    with _initialized_session():
      self.assertAllEqual(expected, embedding_lookup.eval())

  def test_multiple_coalesce(self):
    embedding_dimension = 2
    categorical_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=5)
    with hc.coalesced_hash_table_scope():
      ha = hc.hash_table_column(
          categorical_a, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
      hb = hc.hash_table_column(
          categorical_b, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
    state_manager = _TestStateManager()
    ha.create_state(state_manager)
    hb.create_state(state_manager)
    inputs = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((1, 0), (2, 0), (3, 0), (4, 0)),
            values=np.array((1, 2, 3, 4)),
            dense_shape=(5, 2))
    }
    embedding_a = ha.get_dense_tensor(fc.FeatureTransformationCache(inputs), state_manager)
    embedding_b = hb.get_dense_tensor(fc.FeatureTransformationCache(inputs), state_manager)
    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('CoalescedHashTable/HashTable_0',),
                          tuple([v.name for v in global_vars]))
    expected_a = ((1.,1.),(2.,2.),(0.,0.),(1.,1.))
    expected_b = ((0.,0.),(1.,1.),(1.,1.),(1.,1.),(1.,1.))
    with _initialized_session():
      self.assertAllEqual(expected_a, embedding_a.eval())
      self.assertAllEqual(expected_b, embedding_b.eval())

  def test_different_coalesced_signature(self):
    embedding_dimension = 2
    categorical_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=5)
    with hc.coalesced_hash_table_scope():
      ha = hc.hash_table_column(
          categorical_a, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='mean')
      hb = hc.hash_table_column(
          categorical_b, dimension=embedding_dimension + 1,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
    state_manager = _TestStateManager()
    ha.create_state(state_manager)
    hb.create_state(state_manager)
    inputs = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((1, 0), (2, 0), (3, 0), (4, 0)),
            values=np.array((1, 2, 3, 4)),
            dense_shape=(5, 2))
    }
    embedding_a = ha.get_dense_tensor(fc.FeatureTransformationCache(inputs), state_manager)
    embedding_b = hb.get_dense_tensor(fc.FeatureTransformationCache(inputs), state_manager)
    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('CoalescedHashTable/HashTable_0',
                           'CoalescedHashTable_1/HashTable_0'),
                          tuple([v.name for v in global_vars]))
    expected_a = ((1.,1.),(1.,1.),(0.,0.),(1.,1.))
    expected_b = ((0.,0.,0.),(1.,1.,1.),(1.,1.,1.),(1.,1.,1.),(1.,1.,1.))
    with _initialized_session():
      self.assertAllEqual(expected_a, embedding_a.eval())
      self.assertAllEqual(expected_b, embedding_b.eval())
  '''
  def test_different_runtime_signature_coalesce(self):
    embedding_dimension = 2
    lookup_hook = ReadOnlyHook()
    categorical_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=5)
    with hc.coalesced_hash_table_scope():
      ha = hc.hash_table_column(
          categorical_a, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
      hb = hc.hash_table_column(
          categorical_b, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='sum',
          embedding_lookup_hooks=lookup_hook)

    inputs = _LazyBuilder({
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((1, 0), (2, 0), (3, 0), (4, 0)),
            values=np.array((1, 2, 3, 4)),
            dense_shape=(5, 2))
    })
    embedding_a = ha._get_dense_tensor(inputs)
    embedding_b = hb._get_dense_tensor(inputs)
    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(('CoalescedHashTable/HashTable_0',),
                          tuple([v.name for v in global_vars]))
    expected_a = ((1.,1.),(2.,2.),(0.,0.),(1.,1.))
    expected_b = ((0.,0.),(0.,0.),(0.,0.),(0.,0.),(0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected_a, embedding_a.eval())
      self.assertAllEqual(expected_b, embedding_b.eval())
  '''
  def test_coalesced_dense_features(self):
    embedding_dimension = 2
    categorical_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=5)
    with hc.coalesced_hash_table_scope():
      ha = hc.hash_table_column(
          categorical_a, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='mean')
      hb = hc.hash_table_column(
          categorical_b, dimension=embedding_dimension + 1,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
    
    inputs = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((1, 0), (1, 1), (2, 0), (2, 1)),
            values=np.array((1, 2, 3, 4)),
            dense_shape=(4, 2))
    }
    dense_features = df.DenseFeatures(
        feature_columns=(ha, hb))(inputs)

    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('dense_features/aaa/CoalescedHashTable/HashTable_0',
         'dense_features/bbb/CoalescedHashTable/HashTable_0'),
        tuple([v.name for v in global_vars]))
    expected = (
        (1.,1.,0.,0.,0.),
        (1.,1.,2.,2.,2.),
        (0.,0.,2.,2.,2.),
        (1.,1.,0.,0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected, dense_features)

  def test_shared_coalesced(self):
    embedding_dimension = 2
    categorical_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=5)
    categorical_c = fc.categorical_column_with_identity(
        key='ccc', num_buckets=4)
    with hc.coalesced_hash_table_scope():
      h1 = hc.shared_hash_table_column(
          categorical_a, dimension=embedding_dimension,
          shared_name='shared_name',
          initializer=init_ops.ones_initializer(),
          combiner='mean')
      h2 = hc.hash_table_column(
          categorical_b, dimension=embedding_dimension + 1,
          initializer=init_ops.ones_initializer(),
          combiner='sum')
      h3 = hc.shared_hash_table_column(
          categorical_c, dimension=embedding_dimension,
          shared_name='shared_name',
          initializer=init_ops.ones_initializer(),
          combiner='mean')

    inputs = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((1, 0), (1, 1), (2, 0), (2, 1)),
            values=np.array((1, 2, 3, 4)),
            dense_shape=(4, 2)),
        'ccc': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (0, 1), (2, 0), (2, 1)),
            values=np.array((1, 2, 3, 2)),
            dense_shape=(4, 2))
    }
    dense_features = df.DenseFeatures(
        feature_columns=(h1, h2, h3))(inputs)
    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('dense_features/shared_name/CoalescedHashTable/HashTable_0',
         'dense_features/bbb/CoalescedHashTable/HashTable_0'),
        tuple([v.name for v in global_vars]))
    expected = (
        (1.,1.,0.,0.,0.,1.,1.),
        (1.,1.,2.,2.,2.,0.,0.),
        (0.,0.,2.,2.,2.,1.,1.),
        (1.,1.,0.,0.,0.,0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected, dense_features)

  def test_multiple_coalesced_dense_features(self):
    embedding_dimension = 2
    categorical_a = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_b = fc.categorical_column_with_identity(
        key='bbb', num_buckets=5)
    with hc.coalesced_hash_table_scope():
      ha = hc.hash_table_column(
          categorical_a, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='mean')
      hb = hc.hash_table_column(
          categorical_b, dimension=embedding_dimension,
          initializer=init_ops.ones_initializer(),
          combiner='sum')

    inputs = {
        'aaa': sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=np.array((2, 0, 1, 1)),
            dense_shape=(4, 5)),
        'bbb': sparse_tensor.SparseTensorValue(
            indices=((1, 0), (1, 1), (2, 0), (2, 1)),
            values=np.array((1, 2, 3, 4)),
            dense_shape=(4, 2))
    }
    dense_features1 = df.DenseFeatures(
        feature_columns=(ha,))(inputs)
    dense_features2 = df.DenseFeatures(
        feature_columns=(hb,))(inputs)
    # Assert expected HashTable variable and lookups
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertItemsEqual(
        ('dense_features/aaa/CoalescedHashTable/HashTable_0',),
        tuple([v.name for v in global_vars]))
    expected_1 = ((1.,1.),(1.,1.),(0.,0.),(1.,1.))
    expected_2 = ((0.,0.),(2.,2.),(2.,2.),(0.,0.))
    with _initialized_session():
      self.assertAllEqual(expected_1, dense_features1)
      self.assertAllEqual(expected_2, dense_features2)

  def test_coalesced_save_restore(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    children = ['v1', 'v2', 'v3']
    save_graph = ops.Graph()
    with save_graph.as_default(), self.session(graph=save_graph) as sess:
      ids1 = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      ids2 = constant_op.constant([4, 5, 6], dtype=dtypes.int64) + (1 << 52)
      ids3 = constant_op.constant([7, 8, 9], dtype=dtypes.int64) + (2 << 52)
      ids = array_ops.concat([ids1, ids2, ids3], axis=0)
      ht = hash_table.DistributedHashTable(
          [10], dtypes.float32, init_ops.zeros_initializer(dtypes.float32),
          partitioner=hash_table.FixedSizeHashTablePartitioner(2),
          children=children, name='coalesced')
      var = variables_lib.VariableV1([0], name="variable1", dtype=dtypes.float32)
      loss = math_ops.reduce_sum(array_ops.concat([
          ht.lookup(ids1),
          ht.lookup(ids2) * 2.,
          ht.lookup(ids3) * 3.], axis=1)) + var
      optimizer = adagrad.AdagradOptimizer(0.1)
      train = optimizer.minimize(loss)
      sess.run(variables_lib.global_variables_initializer())
      sess.run(train)
      v1, e1, loss1 = sess.run([var, ht.lookup(ids), loss])
      object_saver = saver_module.Saver(sharded=True)
      save_path = object_saver.save(sess, checkpoint_prefix)

    restore_graph = ops.Graph()
    with restore_graph.as_default(), self.test_session(
        graph=restore_graph) as sess:
      ids1 = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      ids2 = constant_op.constant([4, 5, 6], dtype=dtypes.int64) + (1 << 52)
      ids3 = constant_op.constant([7, 8, 9], dtype=dtypes.int64) + (2 << 52)
      ids = array_ops.concat([ids1, ids2, ids3], axis=0)
      ht = hash_table.DistributedHashTable(
          [10], dtypes.float32, init_ops.zeros_initializer(dtypes.float32),
          partitioner=hash_table.FixedSizeHashTablePartitioner(2),
          children=children, name='coalesced')
      var = variables_lib.VariableV1([0.0], name="variable1")
      loss = math_ops.reduce_sum(array_ops.concat([
          ht.lookup(ids1),
          ht.lookup(ids2) * 2.,
          ht.lookup(ids3) * 3.], axis=1)) + var
      optimizer = adagrad.AdagradOptimizer(0.1)

      object_saver = saver_module.Saver(sharded=True)
      object_saver.restore(sess, save_path)
      v2, e2, loss2 = sess.run([var, ht.lookup(ids), loss])
      self.assertAllEqual(v1, v2)
      self.assertAllEqual(e1, e2)
      self.assertAllEqual(loss1, loss2)

  def test_coalesced_save_normal_restore(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    children = ['v1', 'v2', 'v3']
    save_graph = ops.Graph()
    with save_graph.as_default(), self.session(graph=save_graph) as sess:
      ids1 = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      ids2 = constant_op.constant([4, 5, 6], dtype=dtypes.int64) + (1 << 52)
      ids3 = constant_op.constant([7, 8, 9], dtype=dtypes.int64) + (2 << 52)
      ids = array_ops.concat([ids1, ids2, ids3], axis=0)
      ht = hash_table.DistributedHashTable(
          [10], dtypes.float32, init_ops.zeros_initializer(dtypes.float32),
          partitioner=hash_table.FixedSizeHashTablePartitioner(2),
          children=children, name='coalesced')
      var = variables_lib.VariableV1([0], name="variable1", dtype=dtypes.float32)
      loss = math_ops.reduce_sum(array_ops.concat([
          ht.lookup(ids1),
          ht.lookup(ids2) * 2.,
          ht.lookup(ids3) * 3.], axis=1)) + var
      optimizer = adagrad.AdagradOptimizer(0.1)
      train = optimizer.minimize(loss)
      sess.run(variables_lib.global_variables_initializer())
      sess.run(train)
      v1, e1, loss1 = sess.run([var, ht.lookup(ids), loss])
      object_saver = saver_module.Saver(sharded=True)
      save_path = object_saver.save(sess, checkpoint_prefix)

    restore_graph = ops.Graph()
    with restore_graph.as_default(), self.test_session(
        graph=restore_graph) as sess:
      ht1 = hash_table.DistributedHashTable(
          [10], dtypes.float32, init_ops.zeros_initializer(dtypes.float32),
          partitioner=hash_table.FixedSizeHashTablePartitioner(2),
          name='v1')
      ht2 = hash_table.DistributedHashTable(
          [10], dtypes.float32, init_ops.zeros_initializer(dtypes.float32),
          partitioner=hash_table.FixedSizeHashTablePartitioner(2),
          name='v2')
      ht3 = hash_table.DistributedHashTable(
          [10], dtypes.float32, init_ops.zeros_initializer(dtypes.float32),
          partitioner=hash_table.FixedSizeHashTablePartitioner(2),
          name='v3')
      var = variables_lib.VariableV1([0.0], name="variable1")
      ids1 = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      ids2 = constant_op.constant([4, 5, 6], dtype=dtypes.int64)
      ids3 = constant_op.constant([7, 8, 9], dtype=dtypes.int64)
      loss = math_ops.reduce_sum(array_ops.concat([
          ht1.lookup(ids1),
          ht2.lookup(ids2) * 2.,
          ht3.lookup(ids3) * 3.], axis=1)) + var
      embeddings = array_ops.concat(
          [ht1.lookup(ids1), ht2.lookup(ids2), ht3.lookup(ids3)],
          axis=0)

      object_saver = saver_module.Saver(sharded=True)
      object_saver.restore(sess, save_path)
      v2, e2, loss2 = sess.run([var, embeddings, loss])
      self.assertAllEqual(v1, v2)
      self.assertAllEqual(e1, e2)
      self.assertAllEqual(loss1, loss2)

  def test_coalesced_restore_partial(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    save_graph = ops.Graph()
    bucket_size = 10000
    dimension = 8
    with save_graph.as_default(), self.session(graph=save_graph) as sess:
      categorical_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=bucket_size)
      categorical_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=bucket_size)
      categorical_c = fc.categorical_column_with_identity(
          key='ccc', num_buckets=bucket_size)
      with hc.coalesced_hash_table_scope():
        h1 = hc.hash_table_column(
            categorical_a, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
        h2 = hc.hash_table_column(
            categorical_b, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
        h3 = hc.hash_table_column(
            categorical_c, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
      inputs = {
          'aaa': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((1,2,3), dtype=np.int64),
              dense_shape=(1, 3)),
          'bbb': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((4,5,6), dtype=np.int64),
              dense_shape=(1, 3)),
          'ccc': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((7,8,9), dtype=np.int64),
              dense_shape=(1, 3)),
      }
      dense_features = df.DenseFeatures(
          feature_columns=(h1, h2, h3))(inputs)
      var = variables_lib.VariableV1([0], name="variable1", dtype=dtypes.float32)
      loss = math_ops.reduce_sum(dense_features) + var
      optimizer = adagrad.AdagradOptimizer(0.1)
      train = optimizer.minimize(loss)

      sess.run(variables_lib.global_variables_initializer())
      sess.run(train)

      ids1 = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      ids2 = constant_op.constant([4, 5, 6], dtype=dtypes.int64) + (1 << 52)
      ids3 = constant_op.constant([7, 8, 9], dtype=dtypes.int64) + (2 << 52)
      ids = array_ops.concat([ids1, ids2, ids3], axis=0)

      store = variable_scope._get_default_variable_store()
      ht = store._partitioned_vars['dense_features/aaa/CoalescedHashTable']
      v1, e1, loss1 = sess.run([var, ht.lookup(ids), loss])

      object_saver = saver_module.Saver(sharded=True)
      save_path = object_saver.save(sess, checkpoint_prefix)

    restore_graph = ops.Graph()
    with restore_graph.as_default(), self.test_session(
        graph=restore_graph) as sess:
      categorical_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=bucket_size)
      categorical_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=bucket_size)
      categorical_c = fc.categorical_column_with_identity(
          key='ddd', num_buckets=bucket_size)
      with hc.coalesced_hash_table_scope():
        h1 = hc.hash_table_column(
            categorical_a, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
        h2 = hc.hash_table_column(
            categorical_b, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
        h3 = hc.hash_table_column(
            categorical_c, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
      inputs = {
          'aaa': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((1,2,3), dtype=np.int64),
              dense_shape=(1, 3)),
          'bbb': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((4,5,6), dtype=np.int64),
              dense_shape=(1, 3)),
          'ddd': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((7,8,9), dtype=np.int64),
              dense_shape=(1, 3)),
      }
      dense_features = df.DenseFeatures(
          feature_columns=(h1, h2, h3))(inputs)
      var = variables_lib.VariableV1([0], name="variable1", dtype=dtypes.float32)
      loss = math_ops.reduce_sum(dense_features) + var

      object_saver = saver_module.Saver(sharded=True)
      object_saver.restore(sess, save_path)

      ids1 = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      ids2 = constant_op.constant([4, 5, 6], dtype=dtypes.int64) + (1 << 52)
      ids3 = constant_op.constant([7, 8, 9], dtype=dtypes.int64) + (2 << 52)
      ids = array_ops.concat([ids1, ids2, ids3], axis=0)

      store = variable_scope._get_default_variable_store()
      ht = store._partitioned_vars['dense_features/aaa/CoalescedHashTable']
      v2, e2, loss2 = sess.run([var, ht.lookup(ids), loss])

      self.assertAllEqual(v1, v2)
      self.assertAllEqual(e1[:6], e2[:6])
      self.assertNotEqual(loss1, loss2)

  def test_coalesced_restore_multi_paths(self):
    bucket_size = 10000
    dimension = 8

    # make first checkpoint and save
    checkpoint_directory = LocalTempDir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    g1 = ops.Graph()
    with g1.as_default(), self.session(graph=g1) as sess:
      categorical_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=bucket_size)
      categorical_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=bucket_size)
      with hc.coalesced_hash_table_scope():
        h1 = hc.hash_table_column(
            categorical_a, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
        h2 = hc.hash_table_column(
            categorical_b, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
      inputs = {
          'aaa': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((1,2,3), dtype=np.int64),
              dense_shape=(1, 3)),
          'bbb': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((4,5,6), dtype=np.int64),
              dense_shape=(1, 3)),
      }
      dense_features = df.DenseFeatures(
          feature_columns=(h1, h2))(inputs)
      var = variables_lib.VariableV1([0], name="variable1", dtype=dtypes.float32)
      loss = math_ops.reduce_sum(dense_features) + var
      optimizer = adagrad.AdagradOptimizer(0.1)
      train = optimizer.minimize(loss)

      sess.run(variables_lib.global_variables_initializer())
      sess.run(train)

      ids1 = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      ids2 = constant_op.constant([4, 5, 6], dtype=dtypes.int64) + (1 << 52)

      store = variable_scope._get_default_variable_store()
      ht = store._partitioned_vars['dense_features/aaa/CoalescedHashTable']
      v1, a1, b1 = sess.run([var, ht.lookup(ids1), ht.lookup(ids2)])

      object_saver = saver_module.Saver(sharded=True)
      save_path1 = object_saver.save(sess, checkpoint_prefix)

    # make second checkpoint and save
    checkpoint_directory = LocalTempDir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    g2 = ops.Graph()
    with g2.as_default(), self.session(graph=g2) as sess:
      categorical_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=bucket_size)
      categorical_c = fc.categorical_column_with_identity(
          key='ccc', num_buckets=bucket_size)
      with hc.coalesced_hash_table_scope():
        h1 = hc.hash_table_column(
            categorical_b, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
        h2 = hc.hash_table_column(
            categorical_c, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
      inputs = {
          'bbb': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((104,105,106), dtype=np.int64),
              dense_shape=(1, 3)),
          'ccc': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((7,8,9), dtype=np.int64),
              dense_shape=(1, 3)),
      }
      dense_features = df.DenseFeatures(
          feature_columns=(h1, h2))(inputs)
      var = variables_lib.VariableV1([0], name="variable1", dtype=dtypes.float32)
      loss = math_ops.reduce_sum(dense_features) + var
      optimizer = adagrad.AdagradOptimizer(0.1)
      train = optimizer.minimize(loss)

      sess.run(variables_lib.global_variables_initializer())
      sess.run(train)

      ids1 = constant_op.constant([104, 105, 106], dtype=dtypes.int64)
      ids2 = constant_op.constant([7, 8, 9], dtype=dtypes.int64) + (1 << 52)

      store = variable_scope._get_default_variable_store()
      ht = store._partitioned_vars['dense_features/bbb/CoalescedHashTable']
      v2, b2, c2 = sess.run([var, ht.lookup(ids1), ht.lookup(ids2)])

      object_saver = saver_module.Saver(sharded=True)
      save_path2 = object_saver.save(sess, checkpoint_prefix)

    # restore from first and second checkpoint
    restore_graph = ops.Graph()
    with restore_graph.as_default(), self.test_session(
        graph=restore_graph) as sess:
      categorical_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=bucket_size)
      categorical_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=bucket_size)
      categorical_c = fc.categorical_column_with_identity(
          key='ccc', num_buckets=bucket_size)
      with hc.coalesced_hash_table_scope():
        h1 = hc.hash_table_column(
            categorical_a, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
        h2 = hc.hash_table_column(
            categorical_b, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
        h3 = hc.hash_table_column(
            categorical_c, dimension=8,
            initializer=init_ops.truncated_normal_initializer(0., 0.01),
            combiner='mean')
      inputs = {
          'aaa': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((1,2,3), dtype=np.int64),
              dense_shape=(1, 3)),
          'bbb': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((4,5,6), dtype=np.int64),
              dense_shape=(1, 3)),
          'ccc': sparse_tensor.SparseTensorValue(
              indices=((0,0),(0,1),(0,2)),
              values=np.array((7,8,9), dtype=np.int64),
              dense_shape=(1, 3)),
      }
      dense_features = df.DenseFeatures(
          feature_columns=(h1, h2, h3))(inputs)
      var = variables_lib.VariableV1([0], name="variable1", dtype=dtypes.float32)
      loss = math_ops.reduce_sum(dense_features) + var

      saver1 = saver_module.Saver(sharded=True)
      saver1.restore(sess, save_path1)

      with hash_table.restore_without_clear_scope():
        saver2 = saver_module.Saver(sharded=True)
        saver2.restore(sess, save_path2)

      ids1 = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
      ids2 = constant_op.constant(
          [104, 105, 106], dtype=dtypes.int64) + (1 << 52)
      ids3 = constant_op.constant([7, 8, 9], dtype=dtypes.int64) + (2 << 52)

      store = variable_scope._get_default_variable_store()
      ht = store._partitioned_vars['dense_features/aaa/CoalescedHashTable']
      v3, a3, b3, c3 = sess.run(
          [var, ht.lookup(ids1), ht.lookup(ids2), ht.lookup(ids3)])

      self.assertAllEqual(v3, v2)
      self.assertAllEqual(a3, a1)
      self.assertAllEqual(b3, b2)
      self.assertAllEqual(c3, c2)


class _TestStateManager(fc.StateManager):

  def __init__(self, trainable=True):
    # Dict of feature_column to a dict of variables.
    self._all_variables = collections.defaultdict(lambda: {})
    self._trainable = trainable

  def create_variable(self,
                      feature_column,
                      name,
                      shape,
                      dtype=None,
                      trainable=True,
                      use_resource=True,
                      initializer=None):
    if feature_column not in self._all_variables:
      self._all_variables[feature_column] = {}
    var_dict = self._all_variables[feature_column]
    if name in var_dict:
      return var_dict[name]
    else:
      var = variable_scope.get_variable(
          name=name,
          shape=shape,
          dtype=dtype,
          trainable=self._trainable and trainable,
          use_resource=use_resource,
          initializer=initializer)
      var_dict[name] = var
      return var

  def create_hashtable(self,
                       feature_column,
                       name,
                       shape,
                       dtype,
                       initializer,
                       partitioner,
                       trainable):
    if name in self._all_variables[feature_column]:
      raise ValueError('Variable already exists.')

    table = variable_scope.get_hash_table(
      name, shape, dtype=dtype,
      initializer=initializer,
      partitioner=partitioner,
      trainable=self._trainable and trainable)
    self._all_variables[feature_column][name] = table
    return table

  def get_variable(self, feature_column, name):
    if feature_column not in self._all_variables:
      raise ValueError('Do not recognize FeatureColumn.')
    if name in self._all_variables[feature_column]:
      return self._all_variables[feature_column][name]
    raise ValueError('Could not find variable.')

def _assert_sparse_tensor_value(test_case, expected, actual):
  test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
  test_case.assertAllEqual(expected.indices, actual.indices)
  test_case.assertEqual(
      np.array(expected.values).dtype, np.array(actual.values).dtype)
  test_case.assertAllEqual(expected.values, actual.values)
  test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
  test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)

def _assert_ragged_tensor_value(test_case, expected, actual):
  test_case.assertAllEqual(expected.row_splits.eval(),
                           actual.row_splits.eval())
  test_case.assertAllEqual(expected.values.eval(),
                           actual.values.eval())

if __name__ == '__main__':
  test.main()
