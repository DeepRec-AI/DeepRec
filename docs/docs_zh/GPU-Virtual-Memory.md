# GPU虚拟显存

## 功能介绍

使用 GPU 进行训练时通常会遇到显存不足的问题，该优化在显存不足时使用 cuda 的统一内存地址的 cuMemAllocManaged API 来分配显存，可以使用 CPU 内存来增加显存使用，注意这会造成性能下降。

## 用户接口

该优化默认开启，且只在显存不足时使用，可以使用 `export TF_GPU_VMEM=0` 关闭该优化。注意目前的 GPU Memory Optimization 与GPU虚拟显存优化不兼容，开启 GPU Memory Optimization 时会关闭GPU虚拟显存优化。
