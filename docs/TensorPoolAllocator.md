# TensorPoolAllocator
## 开启 TensorPoolAllocator 优化
TensorPoolAllocator优化目前的DeepRec版本支持单机和分布式的CPU端优化。
存在两个环境变量：`START_STATISTIC_STEP` 和 `STOP_STATISTIC_STEP`，配置开始收集stats的step和结束收集开始优化的step，默认是1000到1100。可以按以下设置减少一开始的冷启动时间。

```bash
export START_STATISTIC_STEP=100
export STOP_STATISTIC_STEP=200
```
一般最少设置开始step为100以去除一开始的初始化图。注意这里的step和训练的step并不一致。
如果初始化图较多，需要相对应提高相应的start和stop step。
大致运行`STOP_STATISTIC_STEP`个step之后可以看到运行时间明显变短。
​

## 使用 jemalloc
设置 `MALLOC` 环境变量后在 python 命令前添加` LD_PRELOAD` jemalloc的动态库即可，比如：
```bash
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:60000,muzzy_decay_ms:60000"
LD_PRELOAD=./libjemalloc.so.2 python ...
```

