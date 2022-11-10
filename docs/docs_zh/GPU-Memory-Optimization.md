# GPU Memory Optimization

## 功能简介

在 GPU 端，TensorFlow 原生 BFCAllocator 存在显存碎片过多的问题，导致显存浪费。该功能能够减少显存碎片，获得比 BFCAllocator 更少的显存占用。该功能在开始运行的前 K 个 step 收集运行信息并使用 cudaMalloc 分配显存，所以会造成性能下降，在收集信息结束后，开启优化，这时的性能会有所提升。

## 用户接口

在 GPU 端，目前的 DeepRec 版本支持单机版本的显存优化，且只针对训练场景。该优化默认关闭，可以使用 `export TF_GPU_ALLOCATOR=tensorpool` 命令开启该优化。注意该优化目前的性能略差于 BFCAllocator，但显存占用比 BFCAllocator 少。比如推荐模型 DBMTL-MMOE 的性能会下降 1% 左右（从 4.72 global_step/sec 降低到 4.67 global_step/sec），但是显存会减少 23% （从 1509.12 MB 减少到 1155.12 MB）。
存在两个环境变量：`START_STATISTIC_STEP` 和 `STOP_STATISTIC_STEP`，配置开始收集stats的step和结束收集开始优化的step，默认是10到110，可以适当调整。



