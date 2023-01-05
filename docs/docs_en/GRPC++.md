# GRPC++
## Introduction
In a large-scale training scenario, users use a large number of workers and ps, resulting in high overhead communication. Tensorflow uses GRPC as a communication protocol, and it is difficult to meet large-scale scenarios (over hundreds or thousands of workers).​

​

To solve above issues, DeepRec provides GRPC++ to support larger-scale training tasks. Based on Sharing-Nothing architecture, BusyPolling, zero copy, Send/Recv Fusion and so on, GRPC++ could greatly improve communication performance and provide much better performance compared with GRPC. GRPC++ supports larger training scale and better training performance, and improves the performance serveral times in some typical business scenarios compared with GRPC.

## User API

Enabling the GRPC++ function is as simple as using GRPC, only need to configure the `Protocol` field.
In some scenarios, especially when there are a lot of Send/Recv operators, configure the `tensor_fuse` field in `config` to enable Send/Recv Ops fusion to avoid too many small packets. **tensor_fuse is disabled by default.**

_tensor_fuse could be used with GRPC as well, which also could brings performance improvement._


### Configure GRPC++

We use seastar as communication framework in GRPC++, and also keep GRPC used by MasterSession. So when enable GRPC++, need to configure another set of ports for seastar. Configure .endpoint_map (filename) in execution directory, content as follows:

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
                         protocol="grpc++")
...

with tf.train.MonitoredTrainingSession(
    master=target,
    config=tf.ConfigProto(tensor_fuse=True, # Enable Send/Recv Ops Fusion
                          ...),
    ) as mon_sess:
  ...
```
### Estimator
Use GRPC++ in Estimator, Need to setup RunConfig with `protocol="grpc++"`:
```python
session_config = tf.ConfigProto(
    tensor_fuse=True, # Enable Send/Recv Ops Fusion
    inter_op_parallelism_threads=16,
    intra_op_parallelism_threads=16)

run_config = tf.estimator.RunConfig(model_dir=model_dir, 
                                    save_summary_steps=train_save_summary_steps,
                                    protocol="grpc++",
                                    session_config=session_config)

...

classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    params={...},
    config=run_config) # run_config
```
_Should not use ParameterServerStrategy, which is not supported by GRPC++._

## Best Practise

In GRPC++, We provide list of enviroments to tune performance.
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
