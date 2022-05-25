# SessionGroup

## 简介
当前Inference场景中，无论用户直接使用TFServing还是使用TF提供的C++接口调用Session::Run，都无法实现多个Session并发处理Request，导致单个Session无法很好的实现CPU or GPU的有效利用。用户如果通过多Instance方式（多进程），无法共享底层的Variable，导致大量使用内存，并且每个Instance各自加载一遍模型，严重影响资源的使用率和模型加载效率。

SessionGroup功能提供了可以配置一组Session，并且将Request通过Round Robin方式分发到某一个Session。SessionGroup中的每个Session有私有的线程池，并且支持每个线程池绑定底层的CPU Core，这样可以最大程度的避免共享资源导致的锁冲突开销。SessionGroup中唯一共享的资源是Variable，所有Session共享底层的Variable，并且模型加载只需要加载一次。

通过使用SessionGroup，可以解决内存占用大，但模型CPU使用率低的问题，大大提高CPU利用率。此外SessionGroup也可以在GPU场景下通过多Session并发执行，大大提高GPU的利用效率。

## 接口介绍

### 直接使用Session::Run场景调用SessionGroup
如果用户直接使用Session::Run方式实现的Serving，可以参考以下使用方式来使用SessionGroup。

#### 创建SessionGroup

session_num指定SessionGroup中创建多少个Session，用户可以通过评估当前单个Session的CPU利用率，判断需要创建多少个Session。比如如果当前单个Session CPU的最高利用率为20%，建议用户配置4个Session。

```c++

TF_RETURN_IF_ERROR(NewSessionGroup(*session_options_,
    session_group, session_num));
TF_RETURN_IF_ERROR((*session_group)->Create(meta_graph_def_.graph_def()));

```

#### Session Run

用户原有代码使用Session::Run可以直接替换为SessionGroup::Run

```c++
status = session_group_->Run(run_options, req.inputs,
    req.output_tensor_names, {}, &resp.outputs, &run_metadata);

```

### TFServing中使用SessionGroup
当前SessionGroup正在支持TFServing（正在开发中）

