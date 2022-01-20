# StarServer
## 简介
随着业务的发展，用户数据量激增，算法模型加宽加深，用户的PS任务规模也随之增大。在超大规模任务场景下(几百甚至上千worker)，原生tensorflow框架中的一些问题被暴露出来，譬如低效的线程池调度，多处关键路径上的锁开销，低效的执行引擎，频繁的rpc带来的开销以及内存使用效率低等等。
​

为了解决用户在超大规模场景下遇到的问题，我们提供了StarServer功能，StarServer对于tensorflow做了全方位的优化，包括graph，线程，executor以及内存等优化。将原有tensorflow中send/recv语义修改为pull/push语义，并且在子图划分上支持了该语义。同时实现了图执行过程中的lock free，大大提高的并发执行子图的效率。StarServer在更大规模的扩展性和性能上优于grpc/grpc++，某些模型是可以成倍的提升性能。StarServer的设计上对PS的runtime进行了优化，整个ps端的图执行实现了无锁化的执行。
## 接口介绍
使用StarServer和GRPC一样，通过简单的配置`protocol`即可。
DeepRec目前支持两个版本的StarServer实现，对应的`protocol`分别是`"star_server"`和`"star_server_lite"`，这两种实现的区别是，`"star_server_lite"`在分图优化上的算法更激进，对于复杂的graph可能会出现分图错误问题，`"star_server"`使用了比较稳健的分图算法。当然`"star_server_lite"`相对`"star_server"`在性能上是有优势的，用户可以按需使用。

### MonitoredTrainingSession训练
```python
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index,
                         protocol="star_server") # 配置 protocol 
...

with tf.train.MonitoredTrainingSession(
    master=target,
    ) as mon_sess:
  ...
```
### Estimator训练
Estimator中使用StarServer，需要通过`RunConfig`来配置`protocol="star_server"`：
```python
session_config = tf.ConfigProto(
    inter_op_parallelism_threads=16,
    intra_op_parallelism_threads=16)

run_config = tf.estimator.RunConfig(model_dir=model_dir, 
                                    save_summary_steps=train_save_summary_steps,
                                    protocol="star_server_lite", # 配置 protocol
                                    session_config=session_config)

...

classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    params={...},
    config=run_config) # 配置 run_config
```
_注意: PS/Worker模式下使用estimator，一定不要使用ParameterServerStrategy。会导致这里的RunConfig的protocol不生效。_
## 最佳实践

