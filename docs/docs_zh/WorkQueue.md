# WorkQueue
## 简介
在大规模分布式异步训练中，如果不同 worker 都读取相同数量的样本，慢节点的训练时间会远长于其他节点，造成长尾现象。随着训练规模的扩大，长尾问题会越来越严重，严重影响了大规模分布式异步训练的整体数据吞吐，拉长了产出模型的时间。
​

我们提供了WorkQueue 类，允许在多种数据源上支持弹性的数据切分，让慢节点训练较少的数据，快节点训练更多的数据，从而大幅缓解了长尾问题的影响，降低了模型训练所需的时间。
​

WorkQueue统一管理所有 worker 上的工作项，各个 worker 在当前剩余的工作项被消费完后会从同一个 WorkQueue 获得新的工作项作为数据源进行训练，从而让训练更快的 worker 获得更多的工作项进行训练。
## 接口介绍
### WorkQueue类介绍
```python
class WorkQueue(works, num_epochs=1,
                shuffle=True,
                seed=None,
                prefix=None,
                num_slices=None,
                name='work_queue')
```
参数的具体含义如下：

- `works`: workers要读的文件的list
- `num_epochs`: 读取全部的数据的次数
- `shuffle`：如果为 True 每个 epoch 都随机重洗数据，否则不进行数据重洗
- `seed`：重洗数据的随机种子，默认为自动
- `prefix`: 工作项（文件名/表名）的前缀，默认为 None, 即无前缀
- `num_slices`: 工作项总数量，集群越不稳定，工作项总数量需要越大，通常为 worker 数量的 10 倍以上，默认为 None 即不分片。读文件的时候num_slices无效。
- `name`: 工作队列的名称
## 方法介绍
### take

method ***WorkQueue.take()*** 

| 作用           | 从全局工作队列获取一个工作项，并下载到本地。 |
| -------------- | -------------------------------------------- |
| **返回值类型** | tensorflow.Tensor                            |
| **参数**       | 无参数                                       |

### input_dataset

method ***WorkQueue.input_dataset()***

| 作用           | 返回一个 Dataset，Dataset的每个元素为一个工作项 |
| -------------- | ----------------------------------------------- |
| **返回值类型** | tensorflow.data.Dataset                         |
| **参数**       | 无参数                                          |

### input_producer
method ***WorkQueue.input_producer()***

| 作用           | 全局工作队列在本地的代理队列，为 Reader 类 Op 使用。 |
| -------------- | ---------------------------------------------------- |
| **返回值类型** | tensorflow.FIFOQueue                                 |
| **参数**       | 无参数                                               |

### add_summary
method ***WorkQueue.add_summary()***

| 作用           | 调用后将会在 tensorboard 中显示 work queue 的水位信息。 |
| -------------- | ------------------------------------------------------- |
| **返回值类型** | 无返回值                                                |
| **参数**       | 无参数                                                  |


## 使用示例
### 使用tf.dataset数据源
```python
from tensorflow.python.ops.work_queue import WorkQueue

# use WorkQueue to allocate tasks
work_queue = WorkQueue([filename, filename1,filename2,filename3])
f_data = work_queue.input_dataset()
# Extract lines from input files using the Dataset API.
dataset = tf.data.TextLineDataset(f_data)
dataset = dataset.shuffle(buffer_size=20000,
                              seed=2021)  # fix seed for reproducing
dataset = dataset.repeat(num_epochs)
dataset = dataset.prefetch(batch_size)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)
```
我们以在model zoo的WDL中使用WorkQueue为例子来展示如何使用workqueue来动态为worker分配的数据。链接：
​

### 文件数据源
```python
from tensorflow.python.ops.work_queue import WorkQueue

work_queue = WorkQueue([path1, path2, path3], shuffle=True)
work_queue.add_summary()
# 创建文件读取器
reader = tf.TextLineReader()
# 从文件列表中读取 2 条记录
keys, values = reader.read_up_to(work_queue.input_producer(), num_records=2)
with tf.train.MonitoredTrainingSession() as sess:
  sess.run(...)
```



### TableRecordReader 数据源

```python
from tensorflow.python.ops.work_queue import WorkQueue

work_queue = WorkQueue(
  [odps_path1, odps_path2, odps_path3], shuffle=True, num_slices=FLAGS.num_workers * 10)
# 创建表读取器
reader = tf.TableRecordReader()
# 从表读取 2 条记录
keys, values = reader.read_up_to(work_queue.input_producer(), num_records=2)
```

