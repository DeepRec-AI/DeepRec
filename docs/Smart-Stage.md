# 自动流水线-SmartStage
## 背景
DeepRec已经提供了stage 功能，该功能可以实现IO Bound操作和计算Bound操作在TensorFlow runtime的驱动下异步执行，从而提高整张图的执行效率。

由于`tf.staged`需要用户指定stage的边界，一方面会增加使用难度，另一方面会导致stage颗粒度不够精细，难以做到更多op的异步执行。因此我们提出了SmartStage功能。用户不需要对TF Graph有OP级别理解的情况下，就可以使stage发挥最大的性能提升。

## 功能说明
在用户的原图中有stage阶段的前提下，通过开启smart stage功能，自动化的寻优最大可以stage的范围，修改实际物理计算图（不影响Graphdef图），从而提高性能。

**注意**：该功能的先决条件是，用户的原图中存在至少一个stage阶段

## 用户接口（CPU 场景）
ConfigProro中定义了如下配置选项

```python
sess_config = tf.ConfigProto()
sess_config.graph_options.optimizer_options.do_smart_stage = True # 通用优化选项
```
## 用户接口（GPU 场景）

ConfigProro中定义了如下配置选项

```python
sess_config = tf.ConfigProto()
sess_config.graph_options.optimizer_options.do_smart_stage_gpu = True # 针对GPU训练优化的选项
```

## 代码示例

```python
import tensorflow as tf

filename_queue = tf.train.string_input_producer(['1.txt'])
reader = tf.TextLineReader()
k, v = reader.read(filename_queue)

var = tf.get_variable("var", shape=[100, 3], initializer=tf.ones_initializer())
v = tf.train.batch([v], batch_size=2, capacity=20 * 3)
v0, v1 = tf.decode_csv(v, record_defaults=[[''], ['']], field_delim=',')
xx = tf.staged([v0, v1])

xx[0]=tf.string_to_hash_bucket(xx[0],num_buckets=10)
xx[0] = tf.nn.embedding_lookup(var, xx[0])
xx[1]=tf.concat([xx[1], ['xxx']], axis = 0)
target = tf.concat([tf.as_string(xx[0]), [xx[1], xx[1]]], 0)

config = tf.ConfigProto()
# enable smart stage
config.graph_options.optimizer_options.do_smart_stage = True
# 对于GPU训练，可以考虑使用以下选项替代do_smart_stage来获得更好地性能
# config.graph_options.optimizer_options.do_smart_stage_gpu = True
# mark target 节点
tf.train.mark_target_node([target])

with tf.train.MonitoredTrainingSession(config=config,
                                       hooks=[tf.make_prefetch_hook()]) as sess:
  for i in range(5):
      print(sess.run([target]))
```
## 性能对比（CPU场景）
在modelzoo中的DLRM模型中测试该功能
机型为Aliyun ECS 实例 ecs.hfg7.8xlarge

- Model name: Intel(R) Xeon(R) Platinum 8369HC CPU @ 3.30GHz
- CPU(s): 32
- Socket(s): 1
- Core(s) per socket: 16
- Thread(s) per core: 2
- Memory: 128G

|      |      case       | global steps/sec |
| :--: | :-------------: | :--------------: |
| DLRM | w/o smart stage |  201 (baseline)  |
| DLRM | w/o smart stage |  212 (+ 1.05x)   |

## 性能对比（GPU场景）

------

在modelzoo中的模型测试该功能在GPU训练场景下的性能。

机器配置：

| CPU  | Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz | 64核心 |
| ---- | --------------------------------------------- | ------ |
| GPU  | NVIDIA A100 80G                               | 单卡   |
| MEM  | 492G                                          |        |

性能结果对比：

| 模型   | 不开启SmartStage <br>(global steps/sec) | do_smartstage <br>(global steps/sec) | do_smartstage_gpu <br>(global steps/sec) |
| ------ | --------------------------------------- | ------------------------------------ | ---------------------------------------- |
| DIEN   | 17.1673                                 | 16.918                               | 17.2557                                  |
| DIN    | 137.584                                 | 132.619                              | 165.069                                  |
| DLRM   | 91.6982                                 | 67.735                               | 188.105                                  |
| DSSM   | 92.4544                                 | 83.7194                              | 101.352                                  |
| DeepFM | 74.7011                                 | 62.1227                              | 93.0858                                  |
