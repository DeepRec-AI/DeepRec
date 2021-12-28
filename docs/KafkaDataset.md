# KafkaDataset

## 功能

1. kafka dataset支持配置多partition，并按时序消费kafka 消息
1. kafka dataset支持保存/恢复状态信息

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
 
kafka_dataset = tf.data.KafkaDataset(topics=["dewu_1_partition:0:0:-1"],
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
