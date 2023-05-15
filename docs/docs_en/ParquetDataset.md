# ParquetDataset

## Description

1. ParquetDataset supports reading data from parquet files.

2. ParquetDataset supports reading parquet files from local or S3/OSS/HDFS file systems.

## UserAPI

### Environment Variable
```python
# If ARROW_NUM_THREADS > 0, specified number of threads will be used.
# If ARROW_NUM_THREADS = 0, no threads will be used.
# If ARROW_NUM_THREADS < 0, all threads will be used.
os.environ['ARROW_NUM_THREADS'] = '2'
```

### ParquetDataset API
```python
class ParquetDataset(dataset_ops.DatasetV2):
  def __init__(
      self, filenames,
      batch_size=1,
      fields=None,
      partition_count=1,
      partition_index=0,
      drop_remainder=False,
      num_parallel_reads=None,
      num_sequential_reads=1):

# Create a `ParquetDataset` from filenames dataset.
def read_parquet(
    batch_size,
    fields=None,
    partition_count=1,
    partition_index=0,
    drop_remainder=False,
    num_parallel_reads=None,
    num_sequential_reads=1):
```

- `filenames`: the filename of parquet file, This parameter can receive the following types.
    - A 0-D or 1-D `tf.string` tensor
    - `string`
    - `list` or `tuple` of `string`
    - `Dataset` containing one or more filenames.

- `batch_size`: *(Optional.)* Maxium number of samples in an output batch.

- `fields`: *(Optional.)* List of DataFrame fields.
    | `filenames` parameter type             | `fields` parameter requirement                     | `fields` parameter type                                                                         |
    |----------------------------------------|----------------------------------------------------|-------------------------------------------------------------------------------------------------|
    | `Tensor`/`DataSet`                     | required                                           | `DataFrame.Field`/`list` or `tuple` of `DataFrame.Field`                                        |
    | `string`/`list` or `tuple` of `string` | optional, the default value means read all columns | `DataFrame.Field`/`list` or `tuple` of `DataFrame.Field`/`string`/`list` or `tuple` of `string` |

- `partition_count`: *(Optional.)* Count of row group partitions.

- `partition_index`: *(Optional.)* Index of row group partitions.

- `drop_remainder`: *(Optional.)* If True, only keep batches with exactly `batch_size` samples.

- `num_parallel_reads`: *(Optional.)* A `tf.int64` scalar representing the number of files to read in parallel. Defaults to reading files sequentially.

- `num_sequential_reads`: *(Optional.)* A `tf.int64` scalar representing the number of batches to read in sequential. Defaults to 1.

### DataFrame
A data frame is a table consisting of multiple named columns. A named column has a logical data type and a physical data type.

#### Logical Type of DataFrame

| Logical Type                            | Output Type                         |
|-----------------------------------------|-------------------------------------|
| Scalar                                  | `tf.Tensor`/`DataFrame.Value`       |
| Fixed-Length List                       | `tf.Tensor`/`DataFrame.Value`       |
| Variable-Length List                    | `tf.SparseTensor`/`DataFrame.Value` |
| Variable-Length Nested List             | `tf.SparseTensor`/`DataFrame.Value` |

#### Physical Type of DataFrame
| Category | Physical Type                                    |
|----------|--------------------------------------------------|
| Integers | `int64` `uint64` `int32` `uint32` `int8` `uint8` |
| Numerics | `float64` `float32` `float16`                    |
| Text     | `string`                                         |

#### DataFrame API

```python
class DataFrame(object):
    class Field(object):
        def __init__(self, name,
            type=None,
            ragged_rank=None,
            shape=None):

    class Value(collections.namedtuple(
        'DataFrameValue', ['values', 'nested_row_splits'])):
        def to_sparse(self, name=None):

# Convert values to tensors or sparse tensors from input dataset.
def to_sparse(num_parallel_calls=None):
```

##### DataFrame.Field API
- `name`: Name of column.
- `type`: data type of column, such as `tf.int64`
- `ragged_rank`: *(optional.)* Specify the number of nesting levels when column is a nested list
- `shape`: *(optional.)* Specify the shape of column when column is a fixed-length list
> Attention: For fix-length list, only the shape needs to be specified.

##### DataFrame.Value Conversion API (Use according to the actual situation)
Since there may be `DataFrame.Value` types in the output of ParquetDataset that cannot be accessed by model directly, it needs to convert the `DataFrame.Value` to SparseTensor. Please use the `to_sparse` API for conversion.
```python
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parquet_dataset_ops
from tensorflow.python.data.experimental.ops import dataframe

ds = parquet_dataset_ops.ParquetDataset(...)
ds.apply(dataframe.to_sparse())
...
```

## Examples

### 1. Read from one file on local filesystem

```python
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parquet_dataset_ops

# Read from a parquet file.
ds = parquet_dataset_ops.ParquetDataset('/path/to/f1.parquet',
                                        batch_size=1024)
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'a': tensora, 'c': tensorc}
```

### 2. Read from filenames dataset

```python
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parquet_dataset_ops

filenames = tf.data.Dataset.from_generator(func, tf.string, tf.TensorShape([]))
# Define data frame fields.
fields = [
    parquet_dataset_ops.DataFrame.Field('A', tf.int64),
    parquet_dataset_ops.DataFrame.Field('C', tf.int64, ragged_rank=1)]
# Read from parquet files by reading upstream filename dataset.
ds = filenames.apply(parquet_dataset_ops.read_parquet(1024, fields=fields))
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'a': tensora, 'c': tensorc}
```

### 3. Read from files on HDFS

```python
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parquet_dataset_ops

# Read from parquet files on remote services for selected fields.
ds = parquet_dataset_ops.ParquetDataset(
    ['hdfs://host:port/path/to/f3.parquet'],
    batch_size=1024,
    fields=['a', 'c'])
ds = ds.prefetch(4)
it = tf.data.make_one_shot_iterator(ds)
batch = it.get_next()
# {'a': tensora, 'c': tensorc}
```
