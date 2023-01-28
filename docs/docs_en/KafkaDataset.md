# KafkaDataset

## Description

1. KafkaDataset supports configuring multiple partitions and consumes kafka messages in time sequence.

2. KafkaDataset supports saving/restoring state information.

## User API

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

- `topics`: A tf.string tensor containing one or more subscriptions, in the format of [topic:partition:offset:length], by default length is -1 for unlimited.
- `servers`: A list of bootstrap servers.
- `group`: The consumer group id.
- `eof`: If True, the kafka reader will stop on EOF.
- `timeout`: The timeout value for the Kafka Consumer to wait (in millisecond).
- `config_global`: A tf.string tensor containing global configuration properties in [Key=Value] format, eg. ["enable.auto.commit=false", "heartbeat.interval.ms=2000"], please refer to 'Global configuration properties' in librdkafka doc.
- `config_topic`: A tf.string tensor containing topic configuration properties in [Key=Value] format, eg. ["auto.offset.reset=earliest"], please refer to 'Topic configuration properties' in librdkafka doc.
- `message_key`: If True, the kafka will output both message value and key.

## Examples

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

---

# KafkaGroupIODataset

## Description

1. KafkaGroupIODataset supports configuring multiple partitions and consumes kafka messages in time sequence.
2. KafkaGroupIODataset supports load balancing within the consumer group.

## User API

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

- `topics`: A `tf.string` tensor containing topic names in [topic] format. For example: ["topic1", "topic2"].
- `group_id`: The id of the consumer group. For example: cgstream.
- `servers`: An optional list of bootstrap servers. For example: `localhost:9092`.
- `stream_timeout`: An optional timeout duration (in milliseconds) to block until the new messages from kafka are fetched. By default it is set to 0 milliseconds and doesn't block for new messages. To block indefinitely, set it to -1.
- `message_poll_timeout`: An optional timeout duration (in milliseconds) after which the kafka consumer throws a timeout error while fetching a single message. This value also represents the intervals at which the kafka topic(s) are polled for new messages while using the `stream_timeout`.
- `configuration`: An optional `tf.string` tensor containing configurations in [Key=Value] format.
  - `Global configuration`: please refer to 'Global configuration properties' in librdkafka doc. Examples include ["enable.auto.commit=false", "heartbeat.interval.ms=2000"]
  - `Topic configuration`: please refer to 'Topic configuration properties' in librdkafka doc. Note all topic configurations should be prefixed with `conf.topic.`. Examples include ["conf.topic.auto.offset.reset=earliest"]
  - `Reference`: [https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md](https://github.com/edenhill/librdkafka/blob/master/CONFIGURATION.md).
- `internal`: Whether the dataset is being created from within the named scope. Default: True.

## Examples

```python

```
