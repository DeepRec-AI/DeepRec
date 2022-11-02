# Embedding Layer Device Placement Optimization

## 背景

在稀疏模型在线GPU推理任务中，模型中Embedding层中只有部分算子被place到GPU上，这导致在推理任务在Embedding层执行时出现大量的CPU-GPU之间的数据拷贝，GPU计算带来的性能提升难以抵消内存拷贝带来的overhead，拖慢推理任务。虽然可以通过实现相关算子的GPU版本进行解决，但是仍然有部分算子由于并行度不高，部分计算必须在CPU上进行等问题导致算子执行效率不高，或者无法实现GPU版本。为此，可以选择将Embedding层place到CPU上执行，从而减少内存拷贝，达到提升性能的目的。

由于用户手动将Embedding层place到CPU上编程较为繁琐，为此我们提出了Embedding Layer Device Placement优化功能。

## 功能说明

Embedding Layer Device Placement优化功能能够自动识别Embedding层，并将其place到CPU上。同时只改变实际计算图，而不影响GraphDef。

## 用户接口（C++）

SessionOptions中提供了如下选项：

```C++
tensorflow::SessionOptions session_options;
session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_embedding_layer_device_placement_optimization(true);
```

