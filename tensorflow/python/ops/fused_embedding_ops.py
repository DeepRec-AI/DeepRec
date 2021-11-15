from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gen_fused_embedding_ops
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_local_sparse_look_up_grad
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_local_sparse_look_up
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_pre_look_up
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_post_look_up
from tensorflow.python.ops.gen_fused_embedding_ops import fused_embedding_sparse_post_look_up_grad
from tensorflow.python.util.tf_export import tf_export


def fused_embedding_lookup_sparse(embedding_weights,
                                  sparse_ids,
                                  combiner=None,
                                  name=None,
                                  max_norm=None):
  if embedding_weights is None:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)
  if isinstance(embedding_weights, variables.PartitionedVariable):
    # get underlying Variables.
    embedding_weights = list(embedding_weights)
  if not isinstance(embedding_weights, list):
    embedding_weights = [embedding_weights]
  if len(embedding_weights) < 1:
    raise ValueError("Missing embedding_weights %s." % embedding_weights)

  with ops.name_scope(name, "fused_embedding_lookup", embedding_weights +
                      [sparse_ids]) as scope:
    if combiner is None:
      logging.warn("The default value of combiner will change from \"mean\" "
                   "to \"sqrtn\" after 2016/11/01.")
      combiner = "mean"
    if combiner not in ("mean", "sqrtn", "sum"):
      raise ValueError("combiner must be one of 'mean', 'sqrtn' or 'sum'")
    if not isinstance(sparse_ids, sparse_tensor.SparseTensor):
      raise TypeError("sparse_ids must be SparseTensor")

    partition_nums = len(embedding_weights)
    if partition_nums == 1:
      emb_vectors = fused_embedding_local_sparse_look_up(sp_values=sparse_ids.values,
                                                         sp_indices=sparse_ids.indices,
                                                         sp_dense_shape=sparse_ids.dense_shape,
                                                         emb_variable=embedding_weights[0],
                                                         T=embedding_weights[0].dtype,
                                                         combiner=combiner,
                                                         max_norm=max_norm)
    else:
      partition_shapes = [w.shape for w in embedding_weights]
      partitioned_values, partitioned_indices = fused_embedding_sparse_pre_look_up(
        partition_shapes=partition_shapes,
        sp_values=sparse_ids.values,
        sp_indices=sparse_ids.indices,
      )
      emb_shards = []
      for i in range(partition_nums):
        embedding = embedding_weights[i]
        sub_partition_values = partitioned_values[i]
        with ops.colocate_with(embedding):
          shard = array_ops.gather(embedding, sub_partition_values)
          emb_shards.append(shard)

      emb_vectors, unused = fused_embedding_sparse_post_look_up(
        emb_shards=emb_shards, partitioned_indices=partitioned_indices,
        sp_dense_shape=sparse_ids.dense_shape,
        partitioned_values=partitioned_values,
        combiner=combiner, max_norm=max_norm
      )
    return emb_vectors


@ops.RegisterGradient("FusedEmbeddingLocalSparseLookUp")
def fused_embedding_local_sparse_look_up_grad(op, top_grad_emb_vec, _):
  grad_sp_values = gen_fused_embedding_ops.fused_embedding_local_sparse_look_up_grad(
    top_grad=top_grad_emb_vec, emb_variable=op.inputs[3],
    sp_values=op.inputs[0], sp_values_offset=op.outputs[1],
    combiner=op.get_attr("combiner"),
    max_norm=op.get_attr("max_norm")
  )
  grads = ops.IndexedSlices(values=grad_sp_values,
                            indices=op.inputs[0])

  return [None, None, None, grads]


@ops.RegisterGradient("FusedEmbeddingSparsePostLookUp")
def fused_embedding_sparse_post_look_up_grad(op, top_grad_emb_vec, _):
  num_partitions = op.get_attr("num_partitions")
  grad_shards = gen_fused_embedding_ops.fused_embedding_sparse_post_look_up_grad(
    top_grad=top_grad_emb_vec, emb_shards=[op.inputs[i] for i in range(0, num_partitions)],
    partitioned_indices=[op.inputs[i] for i in range(num_partitions, 2 * num_partitions)],
    feature_nums=op.outputs[1], combiner=op.get_attr("combiner"),
    max_norm=op.get_attr("max_norm")
  )

  grad_shards = [ops.IndexedSlices(values=grad_shards[i],
                                   indices=op.inputs[i + num_partitions]) for i in range(num_partitions)]

  return grad_shards + [None for _ in range(0, 2 * num_partitions + 1)]
