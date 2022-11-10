# CPU Memory Optimization

## 功能介绍

在 CPU 端，高性能 malloc 库存在大内存分配时带来的 minor pagefault 严重导致的性能问题。该功能能够降低内存的使用量以及 minor pagefault，提升运行性能。
该功能运行时收集内存分配信息（在运行step数达到`START_STATISTIC_STEP`阈值后），然后基于每个运行step的内存分配信息生成分配策略。在生成分配策略时会判断之前生成的内存分配策略是否满足当前的分配需求，如果满足视为一个stable的step，在stable的step数目达到`STABLE_STATISTIC_STEP`阈值或者总收集信息的step数达到`MAX_STATISTIC_STEP`阈值时停止收集内存信息。由于需要收集内存分配信息进行优化，所以在运行一定 step 数之后才能观测到性能提升。

## 用户接口

在 CPU 端，目前的 DeepRec 版本支持单机和分布式的内存优化，该优化默认开启，可以使用 `export ENABLE_MEMORY_OPTIMIZATION=0` 命令关闭该优化。
存在上述提及的几个环境变量：`START_STATISTIC_STEP`，`STABLE_STATISTIC_STEP`和`MAX_STATISTIC_STEP`，配置开始收集stats的step，分配策略稳定分配多少个step后结束，内存分配策略最多运行多少个step后结束。默认值分别为100、10、100。这几个值一般不需要进行改动，初始化图较多时可以调大`START_STATISTIC_STEP`，图比较混乱或者运行的小子图比较多时可以调大`STABLE_STATISTIC_STEP`和`MAX_STATISTIC_STEP`。

### 使用 jemalloc
CPU 端可以搭配 jemalloc 库使用内存优化。设置 `MALLOC` 环境变量后在 python 命令前添加` LD_PRELOAD` jemalloc 的动态库即可，比如：

```bash
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000"
LD_PRELOAD=./libjemalloc.so.2 python ...
```

