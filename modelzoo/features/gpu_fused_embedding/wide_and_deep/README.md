# WDL with GPU Fused Embedding

The model structure, hyper params, dataset, etc, are all same to [WDL](../../../WDL/README.md). Please follow the instruction there to prepare, setup and run the model.

The only difference is that this model use GPU Fused Embedding to acclerate the lookup process. Only change is:

```python
deep_columns.append(tf.feature_column.embedding_column(
    categorical_column,
    dimension=EMBEDDING_DIMENSIONS[column_name],
    combiner='mean', do_fusion=True))
```

## Benchmark

On A100-80GB-PCIE GPU, with 8 cores AMD EPYC 7232P CPU @ 3.20GHz. Average of 5000 iterations. The perf boost:

|         | Avg Time per Iteration |
| ------- | ---------------------- |
| Unfused | 36.38 ms               |
| Fused   | 34.52 ms               |
| SpeedUp | 1.05x                  |