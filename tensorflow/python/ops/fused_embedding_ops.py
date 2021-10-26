from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_fused_embedding_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("fused_embedding_local_sparse_lookup")
def fused_embedding_local_sparse_look_up(
        embeddings,
        sparse_input_tensor,
        combiner="mean",
        max_norm=None):
    return gen_fused_embedding_ops.fused_embedding_local_sparse_look_up(
        sp_values=sparse_input_tensor.values,
        sp_indices=sparse_input_tensor.indices,
        sp_dense_shape=sparse_input_tensor.dense_shape,
        emb_variable=embeddings,
        combiner=combiner,
        max_norm=max_norm
    )


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
