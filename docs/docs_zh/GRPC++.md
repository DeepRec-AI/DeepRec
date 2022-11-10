# GRPC++
## 简介
在大规模训练场景下，用户使用大量worker和ps，导致训练任务通信，数据拷贝等带来很大的开销。原生Tensorflow使用GRPC作为通信协议，在大规模场景下难以满足用户需求的训练性能。
​

针对上述问题，DeepRec提供了GRPC++支持更大规模的训练任务。通过Sharing Nothing架构、BusyPolling机制、用户态零拷贝、Send/Recv融合等多种优化实现，极大的降低了E2E的通信延时，数倍的提高了Server的吞吐能力，从而可以在DeepRec上支持更大的训练规模和更优的训练性能，在一些典型的业务场景上相比较原生的Tensorflow大幅提升了性能。
## 接口介绍
为了方便用户使用，开启GRPC++功能和使用GRPC一样简单，只需要配置`Protocol`字段即可。
在一些场景下，特别是Send/Recv算子特别多时，将Send/Recv算子fuse起来提升性能，具体的是通过在`config`中配置`tensor_fuse`字段来启动此功能，**默认不开启此功能**。

_注：在grpc协议下也能使用tensor_fuse功能。_

### MonitoredTrainingSession训练
```python
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index,
                         protocol="grpc++") # 配置 protocol 
...

with tf.train.MonitoredTrainingSession(
    master=target,
    config=tf.ConfigProto(tensor_fuse=True, # 开启send/recv tensor融合功能
                          ...),
    ) as mon_sess:
  ...
```
### Estimator训练
Estimator中使用GRPC++，需要通过`RunConfig`来配置`protocol="grpc++"`：
```python
session_config = tf.ConfigProto(
    tensor_fuse=True, # 开启send/recv tensor融合功能
    inter_op_parallelism_threads=16,
    intra_op_parallelism_threads=16)

run_config = tf.estimator.RunConfig(model_dir=model_dir, 
                                    save_summary_steps=train_save_summary_steps,
                                    protocol="grpc++", # 配置 protocol
                                    session_config=session_config) # 配置config

...

classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    params={...},
    config=run_config) # 配置 run_config
```
_注意: PS/Worker模式下使用estimator，一定不要使用ParameterServerStrategy。会导致这里的RunConfig的protocol不生效。_
## 最佳实践
GRPC++中除了上述的配置参数之外，还提供了一些环境变量来供用户对性能进行调优。
```python
os.environ['WORKER_ENABLE_POLLING'] = "False"
os.environ['PS_ENABLE_POLLING'] = "False"
```
对于第一组参数：_表示是否需要对通信线程进行polling。_

- `"WORKER_ENABLE_POLLING"`一般都配置成`"False"`。
- `"PS_ENABLE_POLLING"`在比较大的规模下或者混部集群下配置为`"False"`，如果是独立集群或者CPU隔离做的比较好，可以设置为`"True"`。



```python
os.environ['NETWORK_PS_CORE_NUMBER'] = "8"
os.environ['NETWORK_WORKER_CORE_NUMBER'] = "2"
```
对于第二组参数：_表示分配多少个通信线程。为何此处环境变量名是XX_CORE_NUMBER，因为如果通信线程使用polling，那么需要完整占用一个core。_

需要结合任务实际分配的CPU core数量以及是否需要开启上述polling功能来确定。

目前默认值是Min(16, connections)，取最小连接数和16中较小的一个，默认连接数不是最佳的配置。

- 对于worker，一般取2-4。因为PS的数量一般是很少的，总的连接数不会太多，2-4个线程足够使用。
- 对于PS，对于比较大的规模(100-几百数量级别)，一般取8-10足够(假设分配24个CPU core)，具体也需要看模型的计算复杂度以及分配的CPU Core数做相应的微调。对于较小或者较大的规模，相应的数量可以减少或增加。



```python
os.environ["WORKER_DISABLE_PIN_CORES"] = "True"
os.environ["PS_DISABLE_PIN_CORES"] = "True"
```
对于第三组参数：_表示通信线程是否需要绑核。_

DeepRec默认不绑核(此处仅针对GRPC++的通信线程)，用户在独占机器下可以尝试开启此功能。
