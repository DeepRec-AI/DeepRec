# Device Placement Optimization

## Background

In the online GPU inference task of the sparse model, only part of the embedding layer operators are placed on the GPU, which leads to a large amount of data copying between the CPU and GPU when the inference task executes the Embedding layer operators. As a result, the performance improvement brought by GPU computing acceleration is difficult to offset the overhead brought by memory copy, slowing down the inference task. Although it can be solved by implementing the GPU version of related operators, some operators still have problems such as low parallelism and some calculations must be performed on the CPU, resulting in low execution efficiency of the GPU version of the operator, or the GPU version of the operator is difficult to implement. Therefore, users can place the operators of the Embedding layer on the CPU, to reduce memory copies and improve performance.

We propose the Device Placement optimization function, which can automatically place the Embedding Layer on the CPU, thus solving the performance degradation problem caused by the large number of data copy operations between CPU and GPU during the execution of the Embedding Layer.

## Description

The Device Placement optimization function can automatically identify operators in the Embedding layer and place them on the CPU. This function only changes the actual computation graph, not the GraphDef.

## How to use

In user C++ code:

```cpp
tensorflow::SessionOptions session_options;
session_options.config.mutable_graph_options()->mutable_optimizer_options()->set_device_placement_optimization(true);
```
