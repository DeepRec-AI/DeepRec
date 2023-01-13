# GPU Virtual Memory Optimization

## Introduction

Training with GPUs often encounters the problem of insufficient GPU memory, this optimization uses cuda's cuMemAllocManaged API of the uniform memory address to allocate GPU memory when the GPU memory is not enough. This API uses CPU memory to increase GPU memory usage, note that this causes performance degradation.

## User API

This optimization is enabled by default and is only used when there is insufficient GPU memory, and can be turned off using `export TF_GPU_VMEM=0`. Note that the current GPU Memory Optimization is not compatible with GPU Virtual Memory Optimization, and GPU Virtual Memory Optimization will be turned off when GPU Memory Optimization is enabled.
