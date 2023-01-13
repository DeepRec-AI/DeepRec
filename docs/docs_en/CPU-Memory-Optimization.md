# CPU Memory Optimization

## Introduction

On the CPU side, memory libraries ptmalloc or jemalloc can cause severe page faults while allocating large memory chunks common to DL applications. To solve this issue, DeepRec optimizes memory allocation to reduce the memory usage and minor page faults, and improve the running performance. When this optimization is enabled, DeepRec will collect memory allocation information (after the number of steps reaches the `START_STATISTIC_STEP` threshold), and then generate an allocation plan based on the collected memory allocation information of each step. When generating the allocation plan, it will determine whether the previously generated memory allocation plan meets the current allocation requirements, and if it is considered a stable step. When the number of stable steps reaches the `STABLE_STATISTIC_STEP` threshold or the total number of steps collected reaches the `MAX_STATISTIC_STEP` threshold, DeepRec will stop collecting memory information. Since memory allocation information needs to be collected for optimization, the performance gain can only be observed after a certain number of steps.


## User API

On the CPU side, the current version of DeepRec supports the CPU memory optimization of stand-alone and distributed training/inference, which is enabled by default, and can be turned off using the `export ENABLE_MEMORY_OPTIMIZATION=0` command.
There are several environment variables. `START_STATISTIC_STEP` configures the step to start collecting memory information. `STABLE_STATISTIC_STEP` configures how many stable steps the allocation policy ends. `MAX_STATISTIC_STEP` configures the maximal steps to end the memory allocation policy. The default values are 100, 10, and 100, respectively. These values generally do not need to be changed, and the `START_STATISTIC_STEP` can be increased when there are many initialization graphs, and the `STABLE_STATISTIC_STEP` and `MAX_STATISTIC_STEP` can be increased when the main computational graph is irregular or there are more running computational graphs.


### Using jemalloc
The CPU side can adapts the memory optimization with the jemalloc library. After setting the `MALLOC` environment variable, add the `LD_PRELOAD` jemalloc dynamic library before the python command, for example:

```bash
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000"
LD_PRELOAD=./libjemalloc.so.2 python ...
```

