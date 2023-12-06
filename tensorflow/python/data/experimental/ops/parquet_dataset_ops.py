# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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
# =============================================================================
"""Dataset that reads Parquet files. This class is compatible with TensorFlow 1.15."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.util import nest

from tensorflow.python.ops import gen_parquet_ops
from tensorflow.python.data.experimental.ops.parquet_pybind import parquet_fields
from tensorflow.python.data.experimental.ops.parquet_pybind import parquet_filenames_and_fields
from tensorflow.python.data.experimental.ops.dataframe import DataFrame


class DataFrameValueSpec(type_spec.BatchableTypeSpec):
  """A TypeSpec for reading batch of DataFrame.Value from dataset."""

  def value_type(self):
    return DataFrame.Value if self._ragged_rank > 0 else ops.Tensor

  def __init__(self, field):
    """Constructs a type specification for a `tf.RaggedTensor`.

    Args:
      field: The field definition.
    """
    if field.incomplete:
      raise ValueError(
        f'Field {field} is incomplete, please specify dtype and ragged_rank')
    self._field = field

  def _serialize(self):
    return (self._field.dtype, self._field.ragged_rank)

  @property
  def _component_specs(self):
    return self._field.output_specs

  def _to_components(self, value):
    if isinstance(value, DataFrame.Value):
      return [value.values] + list(value.nested_row_splits)
    return [value]

  def _from_components(self, tensor_list):
    if len(tensor_list) < 1:
      return None
    if len(tensor_list) == 1:
      return tensor_list[0]
    return DataFrame.Value(tensor_list[0], tensor_list[1:])

  def _batch(self, batch_size):
    raise NotImplementedError('batching of a bacthed tensor not supported')

  def _unbatch(self):
    raise NotImplementedError('unbatching of a bacthed tensor not supported')

  def _to_legacy_output_types(self):
    return self._field.output_types

  def _to_legacy_output_shapes(self):
    return self._field.output_shapes

  def _to_legacy_output_classes(self):
    return self._field.output_classes


class _ParquetDataset(dataset_ops.DatasetSource):  # pylint: disable=abstract-method
  """A Parquet Dataset that reads batches from parquet files."""

  def __init__(
      self, filename, batch_size, fields,
      partition_count=1,
      partition_index=0,
      drop_remainder=False):
    """Create a `ParquetDataset`.

    Args:
      filename: A 0-D `tf.string` tensor containing one filename.
      batch_size: Maxium number of samples in an output batch.
      fields: List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      drop_remainder: (Optional.) If True, only keep batches with exactly
        `batch_size` samples.
    """
    self._filename = ops.convert_to_tensor(
      filename, dtype=dtypes.string, name='filename')
    self._batch_size = ops.convert_to_tensor(
      batch_size, dtype=dtypes.int64, name='batch_size')
    self._fields = fields
    self._output_specs = {}
    for f in self._fields:
      item = None
      if f.ragged_rank > 0:
        item = DataFrameValueSpec(f)
      else:
        shape = tensor_shape.vector(batch_size if drop_remainder else None)
        if f.shape:
          shape = shape.concatenate(f.shape)
        item = tensor_spec.TensorSpec(shape=shape, dtype=f.dtype)
      self._output_specs[f.name] = item

    self._field_names = nest.flatten({f.name: f.name for f in self._fields})
    self._field_dtypes = nest.flatten({f.name: f.dtype for f in self._fields})
    self._field_ragged_ranks = nest.flatten(
      {f.name: f.ragged_rank for f in self._fields})
    self._partition_count = partition_count
    self._partition_index = partition_index
    self._drop_remainder = drop_remainder

    variant_tensor = gen_parquet_ops.parquet_tabular_dataset_v1(
      self._filename,
      self._batch_size,
      field_names=self._field_names,
      field_dtypes=self._field_dtypes,
      field_ragged_ranks=self._field_ragged_ranks,
      partition_count=self._partition_count,
      partition_index=self._partition_index,
      drop_remainder=self._drop_remainder)
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._output_specs


class ParquetDataset(dataset_ops.DatasetV2):  # pylint: disable=abstract-method
  """A Parquet Dataset that reads batches from parquet files."""

  VERSION = 2002

  @classmethod
  def read_schema(cls, filename, fields=None, lower=False):
    """Read schema from a parquet file.

    Args:
      filename: Path of the parquet file.
      fields: Existing field definitions or field names.
      lower: Convert field name to lower case if not found.

    Returns:
      Field definition list.
    """
    return parquet_fields(filename, fields, lower=lower)

  def __init__(
      self, filenames,
      batch_size=1,
      fields=None,
      partition_count=1,
      partition_index=0,
      drop_remainder=False,
      num_parallel_reads=None,
      num_sequential_reads=1):
    """Create a `ParquetDataset`.

    Args:
      filenames: A 0-D or 1-D `tf.string` tensor containing one or more
        filenames.
      batch_size: (Optional.) Maxium number of samples in an output batch.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      drop_remainder: (Optional.) If True, only keep batches with exactly
        `batch_size` samples.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_sequential_reads: (Optional.) A `tf.int64` scalar representing the
        number of batches to read in sequential. Defaults to 1.
    """
    filenames, self._fields = parquet_filenames_and_fields(filenames, fields)
    self._partition_count = partition_count
    self._partition_index = partition_index
    self._drop_remainder = drop_remainder

    def _create_dataset(f):
      f = ops.convert_to_tensor(f, dtypes.string, name='filename')
      return _ParquetDataset(  # pylint: disable=abstract-class-instantiated
        f, batch_size,
        fields=self._fields,
        partition_count=self._partition_count,
        partition_index=self._partition_index,
        drop_remainder=self._drop_remainder)
    self._impl = self._build_dataset(
      _create_dataset, filenames,
      num_parallel_reads=num_parallel_reads,
      num_sequential_reads=num_sequential_reads)
    super().__init__(self._impl._variant_tensor)  # pylint: disable=protected-access

  @property
  def fields(self):
    return self._fields

  @property
  def partition_count(self):
    return self._partition_count

  @property
  def partition_index(self):
    return self._partition_index

  @property
  def drop_remainder(self):
    return self._drop_remainder

  def _inputs(self):
    return self._impl._inputs()  # pylint: disable=protected-access

  @property
  def element_spec(self):
    return self._impl.element_spec  # pylint: disable=protected-access

  def _build_dataset(
      self, dataset_creator, filenames,
      num_parallel_reads=None,
      num_sequential_reads=1):
    """Internal method to create a `ParquetDataset`."""
    if num_parallel_reads is None:
      return filenames.flat_map(dataset_creator)
    if num_parallel_reads == dataset_ops.AUTOTUNE:
      return filenames.interleave(
        dataset_creator, num_parallel_calls=num_parallel_reads)
    return readers.ParallelInterleaveDataset(
      filenames, dataset_creator,
      cycle_length=num_parallel_reads,
      block_length=num_sequential_reads,
      sloppy=True,
      buffer_output_elements=None,
      prefetch_input_elements=1)


def read_parquet(
    batch_size,
    fields=None,
    partition_count=1,
    partition_index=0,
    drop_remainder=False,
    num_parallel_reads=None,
    num_sequential_reads=1):
  """Create a `ParquetDataset` from filenames dataset.

    Args:
      batch_size: Maxium number of samples in an output batch.
      fields: (Optional.) List of DataFrame fields.
      partition_count: (Optional.) Count of row group partitions.
      partition_index: (Optional.) Index of row group partitions.
      drop_remainder: (Optional.) If True, only keep batches with exactly
        `batch_size` samples.
      num_parallel_reads: (Optional.) A `tf.int64` scalar representing the
        number of files to read in parallel. Defaults to reading files
        sequentially.
      num_sequential_reads: (Optional.) A `tf.int64` scalar representing the
        number of batches to read in sequential. Defaults to 1.
    """
  def _apply_fn(filenames):
    return ParquetDataset(
      filenames,
      batch_size=batch_size,
      fields=fields,
      partition_count=partition_count,
      partition_index=partition_index,
      drop_remainder=drop_remainder,
      num_parallel_reads=num_parallel_reads,
      num_sequential_reads=num_sequential_reads)

  return _apply_fn
