# DeepFM with GPU Fused Embedding

The model structure, hyper params, dataset, etc, are all same to [DLRM](../../../DLRM/README.md). Please follow the instruction there to prepare, setup and run the model.

The only difference is that this model use GPU Fused Embedding to acclerate the lookup process. Only change is:

```python
categorical_embedding_column.use_fused_lookup = True
```

## Benchmark

On A100-80GB-PCIE GPU, with 8 cores AMD EPYC 7232P CPU @ 3.20GHz. Average of 5000 iterations. The perf boost:

|         | Avg Time per Iteration |
| ------- | ---------------------- |
| Unfused | 37.15 ms               |
| Fused   | 31.43 ms               |
| SpeedUp | 1.18x                  |
