# StarServer

## Introduction

In large-scale job (hundreds or even thousands of workers), some problems in the TensorFlow are exposed, such as inefficient thread pool scheduling, lock overhead on multiple critical paths, inefficient execution engine, The overhead caused by frequent rpc and low memory usage efficiency, etc.

In order to solve the problems encountered by users in large-scale scenarios, we provide the StarServer. In StarServer we optimize graph, threadpool, executor, and memory optimization. Change the send/recv semantics in TensorFlow to pull/push semantics, and support this semantics in subgraph division. At the same time, the lock free in the process of graph execution is supported, which greatly improves the concurrent execution efficiency. StarServer perform better than grpc/grpc++ in larger scale job (with thousands of workers). In StarServer we optimizes the runtime of ParameterServer with share-nothing architecture and lock-free graph execution.

## User API

Enabling the StarServer is as simple as using GRPC, only need to configure the `Protocol` field.

In DeepRec there are two StarServer implementation, the `protocol` are `"star_server"` and `"star_server_lite"`. Difference between the two protocol is `"star_server_lite"` use more aggressive graph partition strategy which would bring better performance but probabaly fails in some graph with RNN structure. We suggest `"star_server"` which could support all scenarios.

### Configure StarServer

We use seastar as communication framework in StarServer, and also keep GRPC used by MasterSession. So when enable StarServer, need to configure another set of ports for seastar. Configure .endpoint_map (filename) in execution directory, content as follows:

```
127.0.0.1:3333=127.0.0.1:5555
127.0.0.1:4444=127.0.0.1:6666
```

3333 and 4444 are GRPC ports, and 5555 and 6666 are corresponded seastar ports.

For example, 127.0.0.1:3333 is worker 0 GRPC port, and 127.0.0.1:5555 is worker 0 seastar ports.

### MonitoredTrainingSession

```python
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index,
                         protocol="star_server")
...

with tf.train.MonitoredTrainingSession(
    master=target,
    ) as mon_sess:
  ...
```

### Estimator

Use StarServer in Estimator, Need to setup RunConfig with `protocol="star_server"`:

```python
session_config = tf.ConfigProto(
    inter_op_parallelism_threads=16,
    intra_op_parallelism_threads=16)

run_config = tf.estimator.RunConfig(model_dir=model_dir, 
                                    save_summary_steps=train_save_summary_steps,
                                    protocol="star_server",
                                    session_config=session_config)

...

classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    params={...},
    config=run_config) # 配置 run_config
```
_Should not use ParameterServerStrategy, which is not supported by GRPC++._

## Best Practise

We suggest worker/ps number should be 8:1-10:1 which means 100 workers with 10 ParameterServer when use StarServer. Because StarServer provide powerful ParameterServer, which could support more workers. Besides, we provide list of enviroments to tune performance.


```python
os.environ['WORKER_ENABLE_POLLING'] = "False"
os.environ['PS_ENABLE_POLLING'] = "False"
```
WORKER_ENABLE_POLLING/PS_ENABLE_POLLING true/false means that enable or disable communication threads polling.


```python
os.environ['NETWORK_PS_CORE_NUMBER'] = "8"
os.environ['NETWORK_WORKER_CORE_NUMBER'] = "2"
```

NETWORK_PS_CORE_NUMBER is used to setup Parameter Server communication thread number.
NETWORK_PS_CORE_NUMBER is used to setup Worker communication thread number.

More ParameterServer or Worker need to setup more communication thread number and enable polling.

Currently default communication thread number is Min(16, connections), default thread number is not the best.
- NETWORK_WORKER_CORE_NUMBER is 2-4, because we suggest should not setup too much Parameter Server.
- NETWORK_WORKER_CORE_NUMBER is 8-16, for hundreds of workers, 8-10 thread number is enough(if there is 24 core available). Need to reserve engough cores for compute thread.


```python
os.environ["WORKER_DISABLE_PIN_CORES"] = "True"
os.environ["PS_DISABLE_PIN_CORES"] = "True"
```

WORKER_DISABLE_PIN_CORES: communication threads pin cpu core or not in Worker, not pin core by default.
PS_DISABLE_PIN_CORES: communication threads pin cpu core or not in Parameter Server, not pin core by default.
