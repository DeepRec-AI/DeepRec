# Executor优化
## 功能介绍
Runtime在调度上，做了相应的优化，目前主要优化CPU端执行效率，后续会继续推出GPU runtime优化。

## 用户接口
目前支持三种Executor策略

#### 原生tensorflow executor

默认executor策略。

#### Inline executor

支持Session Run在一个线程上执行完成，减少线程切换，保证在高并发下，达到高吞吐要求，减少框架带来的overhead。一般可以用在serving高并发场景下以及部分PS场景下，能够带来不错的效果。

**使用方式**
```
sess_config = tf.ConfigProto()
sess_config.executor_policy = tf.ExecutorPolicy.USE_INLINE_EXECUTOR

with tf.train.MonitoredTrainingSession(
    master=server.target,
    ...
    config=sess_config) as sess:
  ...
```

#### 基于CostModel的Executor

通过动态Trace指定的Session Run情况，统计与计算多组指标，通过CostModel计算出一个较优的调度策略。该功能中包含了基于关键路径的调度策略和根据CostModel批量执行耗时短的算子的调度策略。

**使用方式**
首先用户可以指定Trace哪些Step的Sesison Run来收集执行指标，默认是收集100～200 Step的指标，通过设置下列环境变量，用户可以自定义此参数。
```
os.environ['START_NODE_STATS_STEP'] = "200"
os.environ['STOP_NODE_STATS_STEP'] = "500"
```
上述示例表示Trace 200～500区间的执行指标。
如果START_NODE_STATS_STEP小于STOP_NODE_STATS_STEP，会Disable此Trace功能，后续CostModel计算也不会被执行。
同时在用户脚本中，需要增加增加下列代码来开启基于CostModel的Executor功能，
```
sess_config = tf.ConfigProto()
sess_config.executor_policy = tf.ExecutorPolicy.USE_COST_MODEL_EXECUTOR

with tf.train.MonitoredTrainingSession(
    master=server.target,
    ...
    config=sess_config) as sess:
  ...
```

