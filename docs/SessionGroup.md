# SessionGroup

## 简介
当前Inference场景中，无论用户直接使用TFServing还是使用TF提供的C++接口调用Session::Run，都无法实现多个Session并发处理Request，导致单个Session无法很好的实现CPU or GPU的有效利用。用户如果通过多Instance方式（多进程），无法共享底层的Variable，导致大量使用内存，并且每个Instance各自加载一遍模型，严重影响资源的使用率和模型加载效率。

SessionGroup功能提供了可以配置一组Session，并且将Request通过Round Robin(支持用户自定义策略)方式分发到某一个Session。SessionGroup中的每个Session有私有的线程池，并且支持每个线程池绑定底层的CPU Core，这样可以最大程度的避免共享资源导致的锁冲突开销。SessionGroup中唯一共享的资源是Variable，所有Session共享底层的Variable，并且模型加载只需要加载一次。

通过使用SessionGroup，可以解决内存占用大，但模型CPU使用率低的问题，大大提高资源利用率，在保证latency的前提下极大提高QPS。此外SessionGroup也可以在GPU场景下通过多Session并发执行，大大提高GPU的利用效率。

## 接口介绍

如果用户使用Tensorflow Serving进行服务，可以使用我们提供的代码: [AlibabaPAI/serving](https://github.com/AlibabaPAI/serving/commits/deeprec)，这里已经提供了接入SessionGroup的功能。也可以使用我们提供的[Processor](https://github.com/alibaba/DeepRec/tree/main/serving)代码，Processor没有提供RPC服务框架，需要用户完善，接入自有RPC框架中。

如果用户使用的是自有seving框架，那么需要做的修改如下。

### 1.直接Session::Run进行serving场景
在Inference场景下，如果用户直接使用Session::Run方式实现的Serving，可以参考以下使用方式来使用SessionGroup。

#### 创建SessionGroup

如果是手动创建Session::Run方式实现的Serving，那么将serving框架中NewSession改为NewSessionGroup。
session_num指定SessionGroup中创建多少个Session，用户可以通过评估当前单个Session的CPU利用率，判断需要创建多少个Session。比如如果当前单个Session CPU的最高利用率为20%，建议用户配置4个Session。

```c++

TF_RETURN_IF_ERROR(NewSessionGroup(*session_options_,
    session_group, session_num));
TF_RETURN_IF_ERROR((*session_group)->Create(meta_graph_def_.graph_def()));

```
参考代码: [Processor](https://github.com/alibaba/DeepRec/blob/main/serving/processor/serving/model_session.cc#L143)

#### SessionGroup Run

用户原有代码使用Session::Run可以直接替换为SessionGroup::Run

```c++
status = session_group_->Run(run_options, req.inputs,
    req.output_tensor_names, {}, &resp.outputs, &run_metadata);

```
参考代码: [Processor](https://github.com/alibaba/DeepRec/blob/main/serving/processor/serving/model_session.cc#L308)

### 2.通过SavedModelBundle进行serving场景
TFServing使用的是SavedModelBundle进行serving的，相关的代码修改参考：[SessionGroup](https://github.com/AlibabaPAI/serving/commit/8b92300da84652f00f13fd20f5df0656cfa26217)，推荐直接使用我们提供的TFServing代码。

#### TFServing中使用SessionGroup

支持SessionGroup的TFServing代码见：[AlibabaPAI/serving](https://github.com/AlibabaPAI/serving/commits/deeprec)

编译文档见：[TFServing编译](https://deeprec.readthedocs.io/zh/latest/TFServing-Compile-And-Install.html)

使用方式如下：
```c++
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --tensorflow_intra_op_parallelism=16 --tensorflow_inter_op_parallelism=16 --use_per_session_threads=true --session_num_per_group=4 --model_base_path=/xxx/pb
```

主要参数：
```c++
session_num_per_group：表示session group中创建几个sessions。

use_per_session_threads：为true表示每个session使用独立的线程池，减少session之间的干扰，建议配置为true。每个session的线程池都是通过tensorflow_intra_op_parallelis和tensorflow_inter_op_parallelism控制大小。
```

