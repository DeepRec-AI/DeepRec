from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
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
                                  fill_empty_row=True,
                                  blocknums=None):

  if sparse_weights is not None:
    raise ValueError("sparse_weights is not supported yet")

  valid_partition_strategy = ['div']
  if partition_strategy not in valid_partition_strategy:
    raise ValueError("{} is not supported yet. Currently only support {}".format(
      partition_strategy, valid_partition_strategy))

  if blocknums is not None:
    raise ValueError("Using blocknums for DynamicEmbeddingVariable is not supported yet")

  if default_id is not None and type(default_id) is not int:
    raise ValueError("default_id must be a integer!")

  partition_nums = len(params)

  if type(params[0]) is EmbeddingVariable:
    if partition_nums != 1:
      raise ValueError("For EmbeddingVariable, do not support partition now")
    # fake shape for now. TBD change in the future
    partition_shapes = [constant([1, 1], dtype=tensorflow.int64)]
  else:
    partition_shapes = [w.shape for w in params]

  with ops.name_scope(name, "fused_embedding_lookup_sparse",
                      params + [sp_ids]) as name:
    prelookup_out = fused_embedding_sparse_pre_look_up(
      partition_shapes=partition_shapes,
        sp_values=sp_ids.values,
        sp_indices=sp_ids.indices,
        sp_dense_shape=sp_ids.dense_shape,
        fill_empty_row=fill_empty_row,
        default_id=default_id,
        prune_invalid_id=bool(prune_invalid_ids),
        partition_strategy=partition_strategy)

    partitioned_values = prelookup_out[0]
    partition_permutations = prelookup_out[1]
    row_empty_and_invalid_flags = prelookup_out[2]
    indices_before_unique = prelookup_out[3]
    unique_idxs = prelookup_out[4]
    unique_counts = prelookup_out[5]
    idx_of_input_to_unique = prelookup_out[6]
    unique_offsets = prelookup_out[7]

    emb_shards = []
    for i in range(partition_nums):
      with ops.colocate_with(params[i]):
        shard = array_ops.gather(params[i], partitioned_values[i])
        emb_shards.append(shard)

    emb_vectors, _ = fused_embedding_sparse_post_look_up(
      emb_shards=emb_shards, partition_permutations=partition_permutations,
      sp_dense_shape=sp_ids.dense_shape,
      indices_before_unique=indices_before_unique,
      row_empty_and_invalid_flags=row_empty_and_invalid_flags,
      unique_counts=unique_counts,
      idx_of_input_to_unique=idx_of_input_to_unique,
      unique_offsets=unique_offsets,
      combiner=combiner, max_norm=max_norm, fill_empty_row=fill_empty_row,
      default_id=default_id
    )

  return emb_vectors


@ops.RegisterGradient("FusedEmbeddingSparsePostLookUp")
def fused_embedding_sparse_post_look_up_gradient(op, top_grad_emb_vec, _):
  num_partitions = op.get_attr("num_partitions")
  combiner = op.get_attr("combiner")
  max_norm = op.get_attr("max_norm")
  fill_empty_row = op.get_attr("fill_empty_row")
  default_id = op.get_attr("default_id")

  emb_shards = [op.inputs[i] for i in range(0, num_partitions)]
  partition_permutations = [op.inputs[i] for i in range(num_partitions, 2 * num_partitions)]
  indices_before_unique = op.inputs[2 * num_partitions + 1]
  row_empty_and_invalid_flags = op.inputs[2 * num_partitions + 2]
  unique_counts = op.inputs[2 * num_partitions + 3]
  idx_of_input_to_unique = op.inputs[2 * num_partitions + 4]
  unique_offsets = op.inputs[2 * num_partitions + 5]

  feature_nums = op.outputs[1]

  grad_shards = fused_embedding_sparse_post_look_up_grad(
    top_grad=top_grad_emb_vec, emb_shards=emb_shards,
    partition_permutations=partition_permutations,
    feature_nums=feature_nums, indices_before_unique=indices_before_unique,
    unique_counts=unique_counts, idx_of_input_to_unique=idx_of_input_to_unique,
    unique_offsets=unique_offsets, row_empty_and_invalid_flags=row_empty_and_invalid_flags,
    combiner=combiner, max_norm=max_norm,
    fill_empty_row=fill_empty_row,
    default_id=default_id
  )
  return grad_shards + [None for _ in range(len(op.inputs) - num_partitions)]
