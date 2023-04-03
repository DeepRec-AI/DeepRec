"""Ops to use variables as resources."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_kv_variable_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

__all__ = ["group_embedding_var_lookup", "_GroupGatherGrad",
           "group_variable_lookup", "_GroupEmbeddingLookup"]

#for GPU EV group_lookup
def group_embedding_var_lookup(params,
                             sp_values,
                             sp_indices,
                             sp_weights,
                             combiners,
                             batch_size,
                             dimensions,
                             ignore_weights,
                             ev_init_value = None):
  if ev_init_value is not None:
    default_value = ev_init_value
    is_use_default_value_tensor = True
  else:
    default_value = ops.convert_to_tensor(1.0)
    is_use_default_value_tensor = False
  if ignore_weights:
    sp_weight = ops.convert_to_tensor(1.0)
    sp_weights = [sp_weight for _ in range(len(sp_values))]
  return gen_kv_variable_ops.group_embedding_var_lookup(params,
                                                        sp_values,
                                                        sp_indices,
                                                        sp_weights,
                                                        batch_size,
                                                        default_value,
                                                        combiners,
                                                        dimensions,
                                                        ignore_weights,
                                                        is_use_default_value_tensor)

@ops.RegisterGradient("GroupEmbeddingVarLookup")
def _GroupGatherGrad(op, *grads):
  ev_num = op.get_attr("num_lookups")
  combiner = op.get_attr("combiner")
  dimension = op.get_attr("dimension")
  return_grads = []
  params = op.inputs[:ev_num]
  sp_indices = op.inputs[ev_num*2:ev_num*3]
  unique_values = op.outputs[ev_num:2*ev_num]
  nnz_grads = gen_kv_variable_ops.group_embedding_variable_lookup_grad(grads[:ev_num],
                                                                      params,
                                                                      unique_values,
                                                                      sp_indices,
                                                                      dimension,
                                                                      combiner)                                                            
  for i in range(ev_num):
    handle = op.inputs[i]
    while handle.op.type != "KvVarHandleOp":
      handle = handle.op.inputs[0]
    params_shape = ops.convert_to_tensor(
        tensor_shape.TensorShape(handle.op.get_attr("shape")))
    indice = unique_values[i]
    grad = nnz_grads[i]
    return_grads.append(ops.IndexedSlices(grad, indice, params_shape))
  for _ in range(ev_num*3 + 2):
    return_grads.append(None)
  # for _ in range(ev_num*4 + 2):
  #   return_grads.append(None)
  return return_grads
  
#for GPU EV group_lookup
def group_variable_lookup(params,
                          sp_values,
                          sp_indices,
                          sp_weights,
                          combiners,
                          batch_size,
                          dimensions,
                          ignore_weights,
                          default_id=None):
  if default_id is not None:
    default_value = default_id
  else:
    default_value = ops.convert_to_tensor(0.0)

  is_use_default_value_tensor = True

  if ignore_weights:
    sp_weight = ops.convert_to_tensor(1.0)
    sp_weights = [sp_weight for _ in range(len(sp_values))]
    
  return gen_kv_variable_ops.group_variable_lookup(params,
                                                  sp_values,
                                                  sp_indices,
                                                  sp_weights,
                                                  batch_size, 
                                                  default_value,
                                                  combiners,
                                                  dimensions,
                                                  ignore_weights=ignore_weights,
                                                  is_use_default_value_tensor=is_use_default_value_tensor)

@ops.RegisterGradient("GroupVariableLookup")
def _GroupEmbeddingLookup(op, *grads):
  ev_num = op.get_attr("num_lookups")
  return_grads = []
  combiner = op.get_attr("combiner")
  dimension = op.get_attr("dimension")
  params = op.inputs[:ev_num]
  unique_values = op.outputs[ev_num:2*ev_num]
  unique_idx = op.outputs[2*ev_num:3*ev_num]
  nnz_grads = gen_kv_variable_ops.group_variable_lookup_grad(grads[:ev_num],
                                                            params,
                                                            unique_values,
                                                            unique_idx,
                                                            dimension,
                                                            combiner)
  for i in range(ev_num):
    params = op.inputs[i]
    with ops.colocate_with(params):
      params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
      params_shape = math_ops.cast(params_shape, dtypes.int32)
    grad = nnz_grads[i]
    indices = unique_values[i]
    size = array_ops.expand_dims(array_ops.size(indices), 0)
    values_shape = array_ops.concat([size, params_shape[1:]], 0)
    values = array_ops.reshape(grad, values_shape)
    
    return_grads.append(ops.IndexedSlices(values, indices, params_shape))
  for _ in range(ev_num*3+2):
    return_grads.append(None)
  # for _ in range(ev_num*4 + 2):
  #   return_grads.append(None)
  return return_grads
