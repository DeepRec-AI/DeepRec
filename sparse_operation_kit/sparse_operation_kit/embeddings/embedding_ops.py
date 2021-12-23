"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sparse_operation_kit import kit_lib
from tensorflow.distribute import MirroredStrategy
from tensorflow.distribute import get_replica_context
from tensorflow.distribute import has_strategy, get_strategy
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops

try:
    import horovod.tensorflow as hvd
except:
    pass

CommToolSet = set(["Strategy", "MPI", "Horovod"])
def get_global_replica_id(comm_tool=None):
    def _strategy():
        replica_ctx = get_replica_context()
        return replica_ctx.replica_id_in_sync_group

    def _MPI():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm.Get_rank()

    def _Horovod():
        return hvd.local_rank()

    if comm_tool is None:
        strategy = get_strategy()
        if strategy:
            return _strategy()
        else:
            try:
                return _MPI()
            except:
                raise RuntimeError("SparseOperationKit can only works with tf.distribute.Strategy or MPI.")
    
    if comm_tool not in CommToolSet:
        raise RuntimeError("SparseOperationKit only works with tf.distribute.Strategy, MPI or Horovod. "+\
                           "But got %s" %comm_tool)

    if "Strategy" == comm_tool:
        return _strategy()
    elif "MPI" == comm_tool:
        return _MPI()
    elif "Horovod" == comm_tool:
        return _Horovod()

def _get_embedding_variable_attr(embedding_variable, attr):
    if not isinstance(attr, str):
        raise ValueError("attr must be a string, but got {}".format(type(attr)))
    if hasattr(embedding_variable, "values"):
        return getattr(embedding_variable.values[0], attr)
    else:
        return getattr(embedding_variable, attr)

def embedding_lookup(embedding_variable, values, training=True, dynamic_input=False, comm_tool=None):
    """
    This function is a wrapper of SOK's dense forward propagation.
    """
    embedding_layer = embedding_variable.embedding_layer

    resource_variable_ops.variable_accessed(embedding_variable)

    return kit_lib.plugin_dense_fprop(embedding_variable._handle,
                                      embedding_layer.handle, 
                                      values=values,
                                      global_replica_id=get_global_replica_id(comm_tool),
                                      training=training,
                                      unique_op_name=embedding_variable.name,
                                      dynamic_input=dynamic_input,
                                      dtype=embedding_variable.dtype)


def embedding_lookup_sparse(embedding_variable, sp_ids, slot_num, training=True, comm_tool=None):
    """
    This function is a wrapper of SOK's sparse forward propagation.
    """
    if not isinstance(sp_ids, sparse_tensor.SparseTensor):
        raise TypeError("sp_ids must be SparseTensor")

    values = sp_ids.values
    indices = check_ops.ensure_shape(sp_ids.indices, shape=(None, 2))
    row_indices = array_ops.transpose(indices, perm=[1, 0])[0]

    embedding_layer = embedding_variable.embedding_layer

    resource_variable_ops.variable_accessed(embedding_variable)

    return kit_lib.plugin_sparse_fprop(embedding_variable._handle,
                                       embedding_layer.handle,
                                       values, row_indices,
                                       get_global_replica_id(comm_tool),
                                       slot_num=slot_num,
                                       training=training,
                                       unique_op_name=embedding_variable.name,
                                       dtype=embedding_variable.dtype)
