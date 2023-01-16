# Executor Optimization

## Introduction
DeepRec introduces some optimizations in runtime scheduling, currently mainly optimizing the execution efficiency of the CPU side, and will continue to launch GPU runtime optimization.

## User API

Three executor policies are currently supported.

#### Native Tensorflow Executor

The default executor policy.

#### Inline Executor

This executor supports the session to execute on one thread, which reduces thread switching and supports high throughput under high concurrency, and reduce the scheduling overhead brought by the framework. It can generally be used in serving high concurrency scenarios and some scenarios that use parameter servers, which can bring good performance.

**Usage**
```
sess_config = tf.ConfigProto()
sess_config.executor_policy = tf.ExecutorPolicy.USE_INLINE_EXECUTOR

with tf.train.MonitoredTrainingSession(
    master=server.target,
    ...
    config=sess_config) as sess:
  ...
```

#### CostModel-based Executor

This executor traces and collects the execution information of the model to build the CostModel and performs optimal execution scheduling based on the CostModel. This executor includes the critical path scheduling and a scheduling strategy for batching execution of operators with short time.

**Usage**

Users can specify which steps to collect execution information. The default is to collect information in 100 to 200 steps. Users can customize both parameters by setting the following environment variables:
```
os.environ['START_NODE_STATS_STEP'] = "200"
os.environ['STOP_NODE_STATS_STEP'] = "500"
```
The above example represents to collect information in 100 to 200 steps.
If the `START_NODE_STATS_STEP` is less than `STOP_NODE_STATS_STEP`, this executor will be disable.
At the same time, in the user script, the following code needs to be added to enable the CostModel-based Executor.
```
sess_config = tf.ConfigProto()
sess_config.executor_policy = tf.ExecutorPolicy.USE_COST_MODEL_EXECUTOR

with tf.train.MonitoredTrainingSession(
    master=server.target,
    ...
    config=sess_config) as sess:
  ...
```

