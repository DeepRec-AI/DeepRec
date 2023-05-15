# ParquetDataset

## 功能

1. parquet dataset支持从parquet文件中读取数据
2. parquet dataset支持从本地以及S3/OSS/HDFS文件系统中读取对应parquet文件

## 接口介绍

### 环境变量介绍

```python
# If ARROW_NUM_THREADS > 0, specified number of threads will be used.
# If ARROW_NUM_THREADS = 0, no threads will be used.
# If ARROW_NUM_THREADS < 0, all threads will be used.
os.environ['ARROW_NUM_THREADS'] = '2'
```

### ParquetDataset接口介绍
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

#### 参数说明

- `filenames`: 文件名，可以接收以下类型的参数。
    - 0-D 或者 1-D 的 `tf.string` 类型 `Tensor`
    - `string` 类型
    - `string` 类型的 `list` 或 `tuple`
    - 包含一个或多个文件名的 `Dataset`

- `batch_size`: *(可选)* 一个输出batch中最大样本数量。

- `fields`: *(可选)* 需要读取的column。

    | filenames 参数类型                     | fields 参数要求               | fields 参数类型要求                                                                             |
    |---------------------------------------|-----------------------------|-----------------------------------------------------------------------------------------------|
    | `Tensor`/`Dataset`                    | 必须传入                      | `DataFrame.Field`/`DataFrame.Field` 类型的`list`或`tuple`                                      |
    | `string`/`string`类型的`list`或`tuple` | 可选, 不传入时默认读取所有column | `DataFrame.Field`/`DataFrame.Field` 类型的`list`或`tuple`/`string`/`string`类型的`list`或`tuple` |


- `partition_count`: *(可选)* row group partitions的数量。

- `partition_index`: *(可选)* row group partitions的索引。

- `drop_remainder`: *(可选)* 如果为`True`, ParquetDataset只会返回大小为`batch_size`的batch，小于`batch_size`的batch将会被丢弃。

- `num_parallel_reads`: *(可选)* `tf.int64`类型的标量，用于设定同时读取的parquet file文件数量。默认逐个依次读取。

- `num_sequential_reads`: *(可选)* `tf.int64`类型的标量，代表按顺序读取的batch数量，默认是1。

### DataFrame介绍

DataFrame是一个包含多个命名的column的表。每一个命名的column都具有一种逻辑类型和一种存储类型。

#### DataFrame支持的逻辑类型

| 逻辑类型                                  | 输出类型                             |
|-----------------------------------------|-------------------------------------|
| 标量(Scalar)                             | `tf.Tensor`/`DataFrame.Value`       |
| 定长List(Fixed-Length List)              | `tf.Tensor`/`DataFrame.Value`       |
| 变长List(Variable-Length List)           | `tf.SparseTensor`/`DataFrame.Value` |
| 变长嵌套List(Variable-Length Nested List) | `tf.SparseTensor`/`DataFrame.Value` |

#### DataFrame支持的存储类型

| 数据分类  | 存储类型                                          |
|---------|--------------------------------------------------|
| 整数     | `int64` `uint64` `int32` `uint32` `int8` `uint8` |
| 浮点数   | `float64` `float32` `float16`                    |
| 文本     | `string`                                         |

#### API 说明

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

##### DataFrame.Field参数说明
- `name`: column 名称
- `type`: 指定元素数据类型，如`tf.int64`
- `ragged_rank`: *(可选)* column为list类型时，用于指定嵌套层数
- `shape`: *(可选)* column为固定shape的list时，用于指定column的shape
> 注：对于固定shape的list (Fix-Length List)，只需要指定shape即可，无需指定ragged_rank。

##### DataFrame.Value转换API (根据实际情况选择使用)
由于ParquetDataset的输出中可能会存在DataFrame.Value, 无法直接接入模型，需要将DataFrame.Value转换为SparseTensor。使用dataset.apply调用to_sparse接口即可完成转换。
```python
import tensorflow as tf
from tensorflow.python.data.experimental.ops import parquet_dataset_ops
from tensorflow.python.data.experimental.ops import dataframe

ds = parquet_dataset_ops.ParquetDataset(...)
ds.apply(dataframe.to_sparse())
...
```

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