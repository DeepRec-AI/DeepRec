# GPU Fused Embedding

As DeepRec's cooperation partner, Nvidia provided fused embedding-related ops to acclerate performance in embedding. Basically, if user is using either

1. `EmbeddingColumn` from `tensorflow/python/feature_column/feature_column_v2.py`
2. `_EmbeddingColumn` from `tensorflow/contrib/layers/python/layers/feature_column.py`

User can turn on `do_fusion` feature by:

```
tf.feature_column.embedding_column(sparse_id_column=xxx, dimension=xxx, do_fusion=True)
```

Then in the embedding lookup process, this embedding column will use Fused GPU embedding ops to acclerate.

## Benchmarks

Please see the READMEs of each model inside this directory.
