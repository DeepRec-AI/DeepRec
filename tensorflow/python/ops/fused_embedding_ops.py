from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_fused_embedding_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("fused_embedding_sparse_lookup")
def fused_embedding_sparse_lookup(
        embeddings,
        sparse_input_tensor,
        combiner="mean"):
    return gen_fused_embedding_ops.fused_embedding_sparse_lookup(
        sp_values=sparse_input_tensor.values,
        sp_indices=sparse_input_tensor.indices,
        sp_dense_shape=sparse_input_tensor.dense_shape,
        emb_variable=embeddings,
        combiner=combiner,
    )


@ops.RegisterGradient("FusedEmbeddingSparseLookUp")
def fused_embedding_sparse_lookup_grad(op, top_grad):
    grad_sp_values, grad_sp_indices = gen_fused_embedding_ops.fused_embedding_sparse_lookup_grad(
        top_grad=top_grad, sp_values=op.input[0], sp_values_offset=op.output[1], combiner=op.get_attr(
            "combiner")
    )
    grads = ops.IndexedSlices(values=grad_sp_values,
                              indices=grad_sp_indices)

    return [None, None, None, grads]
