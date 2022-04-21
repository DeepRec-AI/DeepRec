from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.kv_variable_ops import EmbeddingVariable
from tensorflow.python.ops.gen_fused_embedding_ops import prune_invalid_and_fill_empty_rows
from tensorflow.python.ops.gen_fused_embedding_ops import unique_with_counts_v3
from tensorflow.python.ops.gen_fused_embedding_ops import partition_with_permutation
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_post_look_up_v2
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_post_look_up_v2_grad
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["nn.fused_embedding_lookup_sparse_v2"])
def fused_embedding_lookup_sparse_v2(params,
                                     sp_ids,
                                     sparse_weights=None,
                                     partition_strategy=None,
                                     name=None,
                                     combiner=None,
                                     max_norm=None,
                                     default_id=None,
                                     prune_invalid_ids=False,
                                     fill_empty_row=True,
                                     blocknums=None):

  if sparse_weights is not None:
    raise ValueError("sparse_weights is not supported yet")

  valid_partition_strategy = ['div', 'mod', 'mod_ev']
  if partition_strategy not in valid_partition_strategy:
    raise ValueError("{} is not supported yet. Currently only support {}".format(
      partition_strategy, valid_partition_strategy))

  if blocknums is not None:
    raise ValueError("Using blocknums for DynamicEmbeddingVariable is not supported yet")

  if default_id is not None and type(default_id) is not int:
    raise ValueError("default_id must be a integer!")

  params_white_list = [EmbeddingVariable, ops.Tensor]
  if any([type(param) not in params_white_list for param in params]):
    raise ValueError("Currently fused embedding only support: {}".format(params_white_list))

  partition_nums = len(params)

  if type(params[0]) is EmbeddingVariable:
    partition_strategy = 'mod_ev'
    partition_shapes = [constant([1, 1], dtype=dtypes.int64) for _ in params]  # dummy
  else:
    partition_shapes = [w.shape for w in params]

  with ops.name_scope(name, "fused_embedding_lookup_sparse",
                      params + [sp_ids]) as name:

    sp_values = sp_ids.values
    sp_indices = sp_ids.indices
    sp_dense_shape = sp_ids.dense_shape
    prune_invalid_id = bool(prune_invalid_ids)

    if prune_invalid_ids or fill_empty_row:
      sp_values, sp_indices, row_empty_and_invalid_flags = prune_invalid_and_fill_empty_rows(
        fill_empty_row=fill_empty_row,
        prune_invalid_id=prune_invalid_id,
        default_id=default_id,
        sp_values=sp_values,
        sp_indices=sp_indices,
        sp_dense_shape=sp_dense_shape,
      )
    else:
      row_empty_and_invalid_flags = constant(0, shape=(1, ), dtype=dtypes.int32)  # dummy

    unique_keys, unique_idxs, unique_counts = unique_with_counts_v3(
      input=sp_values,
      CounterType=dtypes.int32,
    )

    if partition_nums > 1:
      partitioned_values, partition_permutation = partition_with_permutation(
        partition_strategy=partition_strategy,
        input=unique_keys,
        partition_shapes=partition_shapes
      )
    else:
      partitioned_values = [unique_keys]
      partition_permutation = constant(0, shape=(1, 1), dtype=dtypes.int32)  # dummy

    emb_shards = []
    for i in range(partition_nums):
      with ops.colocate_with(params[i]):
        shard = array_ops.gather(params[i], partitioned_values[i], counts=unique_counts)
        emb_shards.append(shard)

    emb_vectors, _, _ = fused_embedding_sparse_post_look_up_v2(
      fill_empty_row=fill_empty_row, default_id=default_id,
      combiner=combiner, max_norm=max_norm,
      emb_shards=emb_shards, partition_permutation=partition_permutation,
      sp_dense_shape=sp_dense_shape,
      indices_before_unique=sp_indices,
      row_empty_and_invalid_flags=row_empty_and_invalid_flags,
      unique_idxs=unique_idxs
    )

  return emb_vectors


@ops.RegisterGradient("FusedEmbeddingSparsePostLookUpV2")
def fused_embedding_sparse_post_look_up_v2_gradient(op, top_grad,
                                                 uesless_grad_1, uesless_grad_2):
  num_partitions = op.get_attr("num_partitions")
  combiner = op.get_attr("combiner")
  max_norm = op.get_attr("max_norm")
  fill_empty_row = op.get_attr("fill_empty_row")
  default_id = op.get_attr("default_id")

  emb_shards = [op.inputs[i] for i in range(0, num_partitions)]
  partition_permutation = op.inputs[num_partitions]
  # sp_dense_shape = op.inputs[num_partitions + 1]
  indices_before_unique = op.inputs[num_partitions + 2]
  row_empty_and_invalid_flags = op.inputs[num_partitions + 3]
  unique_idxs = op.inputs[num_partitions + 4]

  feature_nums = op.outputs[1]
  emb_shard_ptrs = op.outputs[2]

  grad_shards = fused_embedding_sparse_post_look_up_v2_grad(
    fill_empty_row=fill_empty_row, default_id=default_id,
    combiner=combiner, max_norm=max_norm,
    top_grad=top_grad, emb_shards=emb_shards,
    emb_shard_ptrs=emb_shard_ptrs,
    partition_permutation=partition_permutation,
    feature_nums=feature_nums, indices_before_unique=indices_before_unique,
    unique_idxs=unique_idxs, row_empty_and_invalid_flags=row_empty_and_invalid_flags)

  return grad_shards + [None for _ in range(len(op.inputs) - num_partitions)]
