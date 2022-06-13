# KafkaDataset

## 功能

1. kafka dataset支持配置多partition，并按时序消费kafka 消息
2. kafka dataset支持保存/恢复状态信息

## 接口介绍

### API说明

```python
class KafkaDataset(dataset_ops.Dataset):
    def __init__(
        self,
        topics,
        servers="localhost",
        group="",
        eof=False,
        timeout=1000,
        config_global=None,
        config_topic=None,
        message_key=False,
)
```

### 参数说明

- topics: A tf.string tensor containing one or more subscriptions, in the format of [topic:partition:offset:length], by default length is -1 for unlimited. 
- servers: A list of bootstrap servers.
- group: The consumer group id.
- eof: If True, the kafka reader will stop on EOF.
- timeout: The timeout value for the Kafka Consumer to wait (in millisecond). 
- config_global: A tf.string tensor containing global configuration properties in [Key=Value] format, eg. ["enable.auto.commit=false", "heartbeat.interval.ms=2000"], please refer to 'Global configuration properties' in librdkafka doc. 
- config_topic: A tf.string tensor containing topic configuration properties in [Key=Value] format, eg. ["auto.offset.reset=earliest"], please refer to 'Topic configuration properties' in librdkafka doc. 
- message_key: If True, the kafka will output both message value and key. 

## 使用示例

```python
import tensorflow as tf
from tensorflow.python.data.ops import iterator_ops
 
kafka_dataset = tf.data.KafkaDataset(topics=["test_1_partition:0:0:-1"],
                                     group="test_group1",
                                     timeout=100,
                                     eof=False)
iterator = iterator_ops.Iterator.from_structure(batch_dataset.output_types)
init_op = iterator.make_initializer(kafka_dataset)
get_next = iterator.get_next()
saveable_obj = tf.data.experimental.make_saveable_from_iterator(iterator)
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable_obj)
saver=tf.train.Saver()
with tf.Session() as sess:
  sess.run(init_op)
  for i in range(100):
    print("Data", sess.run(get_next))
  saver.save(sess, "ckpt/1")
```


# KafkaGroupIODataset


## 功能

1. KafkaGroupIODataset 支持配置多partition，并按时序消费kafka消息
2. KafkaGroupIODataset 支持消费者组内负载均衡

## 接口介绍

### API说明

```python
class KafkaGroupIODataset(dataset_ops.Dataset):
    def __init__(
        self,
        topics,
        group_id,
        servers,
        stream_timeout=0,
        message_poll_timeout=10000,
        configuration=None,
        internal=True,
)
```

### 参数说明

- topics: A `tf.string` tensor containing topic names in [topic] format.
            For example: ["topic1", "topic2"]. 
- group_id: The id of the consumer group. For example: cgstream.
- servers: An optional list of bootstrap servers.
            For example: `localhost:9092`.
- stream_timeout: An optional timeout duration (in milliseconds) to block until the new messages from kafka are fetched. By default it is set to 0 milliseconds and doesn't block for new messages. To block indefinitely, set it to -1.
- message_poll_timeout: An optional timeout duration (in milliseconds) after which the kafka consumer throws a timeout error while fetching a single message. This value also represents the intervals at which the kafka topic(s) are polled for new messages while using the `stream_timeout`. 
- configuration: An optional `tf.string` tensor containing configurations in [Key=Value] format.
  - Global configuration: please refer to 'Global configuration properties' in librdkafka doc. Examples include ["enable.auto.commit=false", "heartbeat.interval.ms=2000"]
  - Topic configuration: please refer to 'Topic configuration properties' in librdkafka doc. Note all topic configurations should be prefixed with `conf.topic.`. Examples include ["conf.topic.auto.offset.reset=earliest"]
  - Reference: https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md. 
- internal: Whether the dataset is being created from within the named scope. Default: True. 

## 使用示例

```python
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.data.ops import readers
import tensorflow as tf


def make_initializable_iterator(ds):
  r"""Wrapper of make_initializable_iterator.
    """
  if hasattr(dataset_ops, 'make_initializable_iterator'):
    return dataset_ops.make_initializable_iterator(ds)
  return ds.make_initializable_iterator()
 
dataset = readers.KafkaGroupIODataset(
        topics=["topic1", "topic2"],
        group_id="cgstream",
        servers="localhost:9092",
        stream_timeout=3000,
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest",
            "enable.auto.commit=true",
        ],
    )

# create the iterators from the dataset
train_iterator = make_initializable_iterator(dataset)
handle = array_ops.placeholder(dtypes.string, shape=[])

iter = iterator_ops.Iterator.from_string_handle(
    handle, train_iterator.output_types, train_iterator.output_shapes,
    train_iterator.output_classes)
next_elements = iter.get_next()

with tf.Session() as sess:
  train_handle = sess.run(train_iterator.string_handle())
  sess.run([train_iterator.initializer])
  for _ in range(100):
    x = sess.run(next_elements, feed_dict={handle: train_handle})
    print(x)
```
