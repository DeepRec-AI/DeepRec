# CUDA Graph Jit Optimization

CUDA Graph是NVIDIA提出的一种CUDA runtime优化，将原本issue到cuda stream上的一系列kernel编译生成一个CUDA Graph的执行对象并issue到选中的CUDA stream上。
后续对该CUDA Graph可执行对象的的调用不需要重复issue的操作，从而节省下了cuda kernel launch的开销。DeepRec目前提供了一种基于Jit自动圈图机制的开启CUDA Graph优化
的方法(JitCugraph)，开启后在推理任务中DeepRec的runtime会自动寻找适合的子图并将其编译封装到CUDA Grahp的可执行文件中，后续执行到该子图时，不需要launch子图中原有的CUDA kernel
而是直接执行已经被缓存在CUDA stream中的该子图对应文件。此功能适用于具有大量小size的GPU算子的场景中，当GPU算子的执行时间和算子的launch开销相当时开启JitCugraph将会有明显的收益。

## 推理任务全局开启JitCugraph

目前JitCugraph支持推理侧全局开启自动圈图的功能，使用时只需要对`SessionOptions`进行配置即可

```cpp
SessionOptions options;
options.config.mutable_gpu_options()->set_cuda_graph_enable_jit(true);
```

- 由于JitGraph在第一遍执行的时候需要对被圈中的子图进行编译, 所以第一次的Serving的latency比较长，建议进行一次预热后再进行性能的测试。
- 目前JitGraph只在单个Session的serving场景下工作，SessionGroup下的用法后续也会支持。
- 目前JitGraph功能和TF XLA有冲突，两者不要同时开启。

## 添加白名单和黑名单

由于JitCugraph对于某些算子上的使用有限制，用户可以通过设置黑名单和白名单环境变量来控制需要进行JitCugraph优化的算子种类。当开启黑名单后，名单内的算子将不会被JitCugraph圈中；当开启白名单后，JitCugraph只会圈中名单内的算子。

```bash
# 使用黑名单
TF_CGMODE_FLAGS="--tf_cgmode_exclude_ops_to_cluster=Relu,Unique"
# 使用白名单
TF_CGMODE_FLAGS="--tf_cgmode_ops_to_cluster=MatMul,AddN"
```

## 控制自动圈图的子图大小

类似TF-XLA, JitCugraph可以通过设置`tf_cgmode_min_cluster_size`和`tf_cgmode_max_cluster_size`两个
环境变量来控制自动子图圈图时的子图包含算子OP的数目
```bash
# 限制最小的子图算子数目为3，包含3个以下OP的子图会被排除(默认值)
TF_CGMODE_FLAGS="--tf_cgmode_min_cluster_size=3"
# 最大cluster size一般不需要设置，默认值为int32最大值，一般认为圈中的子图越大性能越好。 
TF_CGMODE_FLAGS="--tf_cgmode_max_cluster_size=10"
```
注意当使用多个环境变量时用空格隔开
```bash
TF_CGMODE_FLAGS="--tf_cgmode_min_cluster_size=3 --tf_cgmode_exclude_ops_to_cluster=Relu,Unique"
```
