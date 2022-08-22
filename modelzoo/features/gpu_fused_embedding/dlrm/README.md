# DLRM with GPU Fused Embedding

The model structure, hyper params, dataset, etc, are all same to [DLRM](../../../DLRM/README.md). Please follow the instruction there to prepare, setup and run the model.

The only difference is that this model use GPU Fused Embedding to acclerate the lookup process. Only change is:

```python
categorical_embedding_column = tf.feature_column.embedding_column(
    categorical_column, dimension=16, combiner='mean',
    do_fusion=True)
```

## Benchmark

On A100-80GB-PCIE GPU, with 8 cores AMD EPYC 7232P CPU @ 3.20GHz. Average of 5000 iterations. The perf boost:

|         | Avg Time per Iteration |
| ------- | ---------------------- |
| Unfused | 20.78 ms               |
| Fused   | 17.41 ms               |
| SpeedUp | 1.19x                  |
