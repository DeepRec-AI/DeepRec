"""Ops to use variables as resources."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_kv_variable_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

__all__ = ["multi_kv_resource_gather", "_GroupGatherGrad",
           "multi_embedding_sparse_look_up", "_GroupEmbeddingLookup"]

#for GPU EV group_lookup
def multi_kv_resource_gather(params,
                             sp_values,
                             sp_indices,
                             sp_dense_shape,
                             combiners,
                             dimensions,
                             ev_init_value = None):
  if ev_init_value is not None:
    default_value = ev_init_value
    is_use_default_value_tensor = True
  else:
    default_value = ops.convert_to_tensor(1.0)
    is_use_default_value_tensor = False
  return gen_kv_variable_ops.multi_kv_resource_gather(params,
                                                      sp_values,
                                                      sp_indices,
                                                      sp_dense_shape,
                                                      default_value,
                                                      combiners,
                                                      dimensions,
                                                      is_use_default_value_tensor)

@ops.RegisterGradient("MultiKvResourceGather")
def _GroupGatherGrad(op, *grads):
  ev_num = op.get_attr("num_lookups")
  return_grads = []
  combiner = op.get_attr("combiner")
  dimension = op.get_attr("dimension")
  params = op.inputs[:ev_num]
  sp_values = op.inputs[ev_num:2*ev_num]
  sp_values_offset = op.outputs[ev_num:2*ev_num]
  tmp_grads = gen_kv_variable_ops.multi_kv_resource_gather_grad(grads[:ev_num],
                                                                params,
                                                                sp_values,
                                                                sp_values_offset,
                                                                dimension,
                                                                combiner)                                                            
  for i in range(ev_num):
    handle = op.inputs[i]
    while handle.op.type != "KvVarHandleOp":
      handle = handle.op.inputs[0]
    params_shape = ops.convert_to_tensor(
        tensor_shape.TensorShape(handle.op.get_attr("shape")))
    indice = op.inputs[ev_num+i]
    grad = tmp_grads[i]
    size = array_ops.expand_dims(array_ops.size(indice), 0)
    values_shape = array_ops.concat([size, params_shape[0:]], 0)
    grad = array_ops.reshape(grad, values_shape)
    indice = array_ops.reshape(indice, size)
    return_grads.append(ops.IndexedSlices(grad, indice, params_shape))
  for _ in range(ev_num*3 + 1):
    return_grads.append(None)
  return return_grads
  
#for GPU EV group_lookup
def multi_embedding_sparse_look_up(params,
                                  sp_values,
                                  sp_indices,
                                  sp_dense_shape,
                                  combiners,
                                  dimensions):
  return gen_kv_variable_ops.multi_embedding_sparse_look_up(params,
                                                      sp_values,
                                                      sp_indices,
                                                      sp_dense_shape,
                                                      combiners,
                                                      dimensions)

@ops.RegisterGradient("MultiEmbeddingSparseLookUp")
def _GroupEmbeddingLookup(op, *grads):
  ev_num = op.get_attr("num_lookups")
  return_grads = []
  combiner = op.get_attr("combiner")
  dimension = op.get_attr("dimension")
  params = op.inputs[:ev_num]
  sp_values = op.inputs[ev_num:2*ev_num]
  sp_values_offset = op.outputs[ev_num:2*ev_num]
  tmp_grads = gen_kv_variable_ops.multi_embedding_sparse_look_up_grad(grads[:ev_num],
                                                                      params,
                                                                      sp_values,
                                                                      sp_values_offset,
                                                                      dimension,
                                                                      combiner)
  for i in range(ev_num):
    params = op.inputs[i]
    with ops.colocate_with(params):
      params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
      params_shape = math_ops.cast(params_shape, dtypes.int32)
    indice = op.inputs[ev_num+i]
    grad = tmp_grads[i]
    size = array_ops.expand_dims(array_ops.size(indice), 0)
    values_shape = array_ops.concat([size, params_shape[1:]], 0)
    grad = array_ops.reshape(grad, values_shape)
    indice = array_ops.reshape(indice, size)
    return_grads.append(ops.IndexedSlices(grad, indice, params_shape))
  for _ in range(ev_num*3):
    return_grads.append(None)
  return return_grads
