# GPU Memory Optimization

## Introduction

On the GPU side, BFCAllocator, the GPU allocator of TensorFlow, has the problem of excessive memory fragmentation, resulting in wasted GPU memory. This optimization reduces memory fragmentation and results in a smaller memory footprint than BFCAllocator. This optimization collects memory allocation information and uses cudaMalloc to allocate GPU memory in the first K steps, so the performance will be degraded. After the information is collected, the optimization is enabled, and the performance will be improved.

## User API

On the GPU side, the current version of DeepRec only supports the memory optimization of the stand-alone training. This optimization is turned off by default and can be turned on using the `export TF_GPU_ALLOCATOR=tensorpool` command. Note that this optimization currently performs slightly worse than BFCAllocator, but uses less memory than BFCAllocator. For example, the execution performance of the recommended model DBMTL-MMOE will decrease by about 1% (from 4.72 global_step/sec to 4.67 global_step/sec), but the GPU memory will be reduced by 23% (from 1509.12 MB to 1155.12 MB).
There are two environment variables: `START_STATISTIC_STEP` and `STOP_STATISTIC_STEP` to configure the step to start collecting memory allocation information and the step to end collection and start optimizing, respectively. The default value are 10 and 110, respectively, which can be adjusted appropriately.



