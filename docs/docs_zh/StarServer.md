# StarServer
## 简介
随着业务的发展，用户数据量激增，算法模型加宽加深，用户的PS任务规模也随之增大。在超大规模任务场景下(几百甚至上千worker)，原生tensorflow框架中的一些问题被暴露出来，譬如低效的线程池调度，多处关键路径上的锁开销，低效的执行引擎，频繁的rpc带来的开销以及内存使用效率低等等。

为了解决用户在超大规模场景下遇到的问题，我们提供了StarServer功能，StarServer对于tensorflow做了全方位的优化，包括graph，线程，executor以及内存等优化。将原有tensorflow中send/recv语义修改为pull/push语义，并且在子图划分上支持了该语义。同时实现了图执行过程中的lock free，大大提高的并发执行子图的效率。StarServer在更大规模的扩展性和性能上优于grpc/grpc++，某些模型是可以成倍的提升性能。StarServer的设计上对PS的runtime进行了优化，整个ps端的图执行实现了无锁化的执行。

## 接口介绍
使用StarServer和GRPC一样，通过简单的配置`protocol`即可。
DeepRec目前支持两个版本的StarServer实现，对应的`protocol`分别是`"star_server"`和`"star_server_lite"`，这两种实现的区别是，`"star_server_lite"`在分图优化上的算法更激进，对于复杂的graph可能会出现分图错误问题，`"star_server"`使用了比较稳健的分图算法。当然`"star_server_lite"`相对`"star_server"`在性能上是有优势的，用户可以按需使用。

### Configure StarServer
StarServer使用了seastar做为底层的通信库，同时保留了GRPC的接口连接（用于MasterSession），这样需要为seastar配置一组ports。使用StarServer需要在执行目录下配置.endpoint_map文件，格式如下：

```
127.0.0.1:3333=127.0.0.1:5555
127.0.0.1:4444=127.0.0.1:6666
```
其中worker0的GRPC ip/port为127.0.0.1:3333，那么配置的对应节点的seastar port为5555的配置方法为127.0.0.1:3333=127.0.0.1:5555
其中ps0的GRPC ip/port为127.0.0.1:4444，那么配置的对应节点的seastar port为6666的配置方法为127.0.0.1:3333=127.0.0.1:6666

对应的TF_CONFIG的配置中仍然使用的是GRPC的ip/port进行描述。

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

我们建议，在使用StarServer时，worker/ps数量比例为10:1或者8:1左右，由于StarServer提供了强大的ParameterServer，并不需要配置太多数量的ParameterServer即可达到很高的性能。StarServer中除了上述的配置参数之外，还提供了一些环境变量来供用户对性能进行调优。

```python
os.environ['WORKER_ENABLE_POLLING'] = "False"
os.environ['PS_ENABLE_POLLING'] = "False"
```
对于第一组参数：_表示是否需要对通信线程进行polling。_

- `"WORKER_ENABLE_POLLING"`一般都配置成`"False"`。
- `"PS_ENABLE_POLLING"`在混部集群下配置为`"False"`，如果是独立集群或者CPU隔离做的比较好，可以设置为`"True"`。


```python
os.environ['NETWORK_PS_CORE_NUMBER'] = "8"
os.environ['NETWORK_WORKER_CORE_NUMBER'] = "2"
```
对于第二组参数：_表示分配多少个通信线程。为何此处环境变量名是XX_CORE_NUMBER，因为如果通信线程使用polling，那么需要完整占用一个core。_

需要结合任务实际分配的CPU core数量以及是否需要开启上述polling功能来确定。

目前NETWORK_PS_CORE_NUMBER的默认值是Min(8, connections), 取最小连接数和8中较小的一个，默认连接数不是最佳的配置，建议调整该参数。
目前NETWORK_WORKER_CORE_NUMBER的默认值是Min(2, connections), 取最小连接数和2中较小的一个，默认连接数不是最佳的配置，建议调整该参数。

- 对于worker，一般取2-4。因为PS的数量一般是很少的，总的连接数不会太多，2-4个线程足够使用。
- 对于PS，对于比较大的规模(100-几百数量级别)，一般取8-10足够(假设分配24个CPU core)，具体也需要看模型的计算复杂度以及分配的CPU Core数做相应的微调。对于较小或者较大的规模，相应的数量可以减少或增加。


```python
os.environ["WORKER_DISABLE_PIN_CORES"] = "True"
os.environ["PS_DISABLE_PIN_CORES"] = "True"
```
对于第三组参数：_表示通信线程是否需要绑核。_

DeepRec默认不绑核(此处仅针对seastar的通信线程)，用户在独占机器下可以尝试开启此功能。
