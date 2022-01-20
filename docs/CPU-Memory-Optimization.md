# CPU Memory Optimization

## 功能介绍

在 CPU 端，高性能 malloc 库存在大内存分配时带来的 minor pagefault 严重导致的性能问题。该功能能够降低内存的使用量以及 minor pagefault，提升运行性能。
该功能运行时收集信息然后进行优化，所以在运行一定 step 数之后才能观测到性能提升。

## 用户接口

在 CPU 端，目前的 DeepRec 版本支持单机和分布式的内存优化，该优化默认开启，可以使用 `export ENABLE_MEMORY_OPTIMIZATION=0` 命令关闭该优化。
存在两个环境变量：`START_STATISTIC_STEP` 和 `STOP_STATISTIC_STEP`，配置开始收集stats的step和结束收集开始优化的step，默认是1000到1100。可以按以下设置减少一开始的冷启动时间。

```bash
export START_STATISTIC_STEP=100
export STOP_STATISTIC_STEP=200
```

一般最少设置开始 step 为100以去除一开始的初始化图。注意这里的 step 和训练的 step 并不一致。
如果初始化图较多，需要相对应提高相应的 start 和 stop step。
大致运行 `STOP_STATISTIC_STEP` 个step之后可以看到运行时间明显变短。

### 使用 jemalloc
CPU 端可以搭配 jemalloc 库使用内存优化。设置 `MALLOC` 环境变量后在 python 命令前添加` LD_PRELOAD` jemalloc 的动态库即可，比如：

```bash
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000"
LD_PRELOAD=./libjemalloc.so.2 python ...
```

