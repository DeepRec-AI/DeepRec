# ParquetDataset

## 功能

1. parquet dataset支持从parquet文件中读取数据
2. parquet dataset支持从本地以及S3/OSS/HDFS文件系统中读取对应parquet文件

## 接口介绍

### API说明

```python
# If ARROW_NUM_THREADS > 0, specified number of threads will be used.
# If ARROW_NUM_THREADS = 0, no threads will be used.
# If ARROW_NUM_THREADS < 0, all threads will be used.
os.environ['ARROW_NUM_THREADS'] = '2'
```

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

### 参数说明

- filenames: A 0-D or 1-D `tf.string` tensor containing one or more filenames.
- batch_size: (Optional.) Maxium number of samples in an output batch.
- fields: (Optional.) List of DataFrame fields.
- partition_count: (Optional.) Count of row group partitions.
- partition_index: (Optional.) Index of row group partitions.
- drop_remainder: (Optional.) If True, only keep batches with exactly `batch_size` samples.
- num_parallel_reads: (Optional.) A `tf.int64` scalar representing the number of files to read in parallel. Defaults to reading files sequentially.
- num_sequential_reads: (Optional.) A `tf.int64` scalar representing the number of batches to read in sequential. Defaults to 1.

## 使用示例

### 1. Example: Read from one file on local filesystem
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
### 2. Example: Read from filenames dataset
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
...
```
### 3. Example: Read from files on HDFS

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
...
```