from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_fused_embedding_ops
from tensorflow.python.ops.kv_variable_ops import EmbeddingVariable
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_pre_look_up
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_post_look_up
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_post_look_up_grad
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["nn.fused_embedding_lookup_sparse"])
def fused_embedding_lookup_sparse(params,
                                  sp_ids,
                                  sparse_weights=None,
                                  partition_strategy=None,
                                  name=None,
                                  combiner=None,
                                  max_norm=None,
                                  default_id=None,
                                  prune_invalid_ids=False,
                                  blocknums=None):
  if sparse_weights is not None:
    raise ValueError("sparse_weights is not supported yet")

  valid_partition_strategy = ['div']
  if partition_strategy not in valid_partition_strategy:
    raise ValueError("{} is not supported yet. Currently only support {}".format(
      partition_strategy, valid_partition_strategy))

  if default_id is not None and type(default_id) is not int:
    raise ValueError("default_id must be a integer!")

  if blocknums is not None:
    raise ValueError("Using blocknums for DynamicEmbeddingVariable is not supported yet")

  partition_nums = len(params)

  if type(params[0]) is EmbeddingVariable:
    if partition_nums != 1:
      raise ValueError("For EmbeddingVariable, do not support partition now")
    # fake shape for now. TBD change in the future
    partition_shapes = [constant([1, 1], dtype=dtypes.int64)]
  else:
    partition_shapes = [w.shape for w in params]

  with ops.name_scope(name, "fused_embedding_lookup_sparse",
                      params + [sp_ids]) as name:
    partitioned_values, partitioned_indices, \
      row_empty_and_invalid_flags = fused_embedding_sparse_pre_look_up(
          partition_shapes=partition_shapes,
          sp_values=sp_ids.values,
          sp_indices=sp_ids.indices,
          sp_dense_shape=sp_ids.dense_shape,
          fill_empty_row=True,
          default_id=default_id,
          prune_invalid_id=bool(prune_invalid_ids)
      )

    # fixme(marvin): ple align the meaning between pre & post op.
    default_id = 0 if default_id is None else default_id

    emb_shards = []
    for i in range(partition_nums):
      param = params[i]
      sub_partition_values = partitioned_values[i]
      with ops.colocate_with(param):
        shard = array_ops.gather(param, sub_partition_values)
        emb_shards.append(shard)
    emb_vectors, _ = fused_embedding_sparse_post_look_up(
      emb_shards=emb_shards, partitioned_indices=partitioned_indices,
      sp_dense_shape=sp_ids.dense_shape,
      row_empty_and_invalid_flags=row_empty_and_invalid_flags,
      partitioned_values=partitioned_values,
      combiner=combiner, max_norm=max_norm, default_id=default_id
    )
  ops.add_to_collections(ops.GraphKeys.ASYNC_EMBEDDING_OUTPUT_TENSORS, emb_vectors)
  return emb_vectors


@ops.RegisterGradient("FusedEmbeddingSparsePostLookUp")
def fused_embedding_sparse_post_look_up_grad(op, top_grad_emb_vec, _):
  num_partitions = op.get_attr("num_partitions")
  emb_shards = [op.inputs[i] for i in range(0, num_partitions)]
  partitioned_indices = [op.inputs[i] for i in range(num_partitions, 2 * num_partitions)]
  feature_nums = op.outputs[1]
  row_empty_and_invalid_flags = op.inputs[2 * num_partitions + 1]

  grad_shards = gen_fused_embedding_ops.fused_embedding_sparse_post_look_up_grad(
    top_grad=top_grad_emb_vec, emb_shards=emb_shards,
    partitioned_indices=partitioned_indices,
    feature_nums=feature_nums, row_empty_and_invalid_flags=row_empty_and_invalid_flags,
    combiner=op.get_attr("combiner"), max_norm=op.get_attr("max_norm"),
    default_id=op.get_attr("default_id")
  )
  return grad_shards + [None for _ in range(0, 2 * num_partitions + 2)]
