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

from tensorflow.python.framework import load_library, ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow import __version__ as tf_version
if tf_version.startswith("2"):
    using_tf2 = True
elif tf_version.startswith("1"):
    using_tf2 = False
else:
    raise RuntimeError(f"Not supported TF version: {tf_version}")

import os

def in_tensorflow2():
    """
    This function will tell whether the installed TensorFlow is 2.x
    """
    return using_tf2

lib_name = r"libsparse_operation_kit.so"

paths = [r"/usr/local/lib"]

lib_file = None
for path in paths:
    try:
        file = open(os.path.join(path, lib_name))
        file.close()
        lib_file = os.path.join(path, lib_name)
        break
    except FileNotFoundError:
        continue

if lib_file is None:
    raise FileNotFoundError("Could not find %s" %lib_name)

kit_ops = load_library.load_op_library(lib_file)

# for op in dir(kit_ops):
    # print(op)

test = kit_ops.test
get_nccl_unique_id = kit_ops.get_nccl_unique_id
gen_random_seed = kit_ops.gen_random_seed
plugin_init = kit_ops.plugin_init
create_var = kit_ops.create_var
create_embedding_sparse = kit_ops.create_embedding_sparse
create_embedding_dense = kit_ops.create_embedding_dense
plugin_sparse_fprop = kit_ops.plugin_sparse_fprop
plugin_dense_fprop = kit_ops.plugin_dense_fprop
plugin_bprop = kit_ops.plugin_bprop

dump_to_file = kit_ops.dump_to_file
restore_from_file = kit_ops.restore_from_file
load_embedding_values = kit_ops.load_embedding_values
read_embedding_variable = kit_ops.read_embedding_variable_op
if not in_tensorflow2():
    assign_embedding_variable = kit_ops.assign_embedding_variable
    optimizer_init = kit_ops.optimizer_init
    embedding_variable_assign_sub = kit_ops.embedding_variable_assign_sub

create_global_adam_optimizer = kit_ops.create_global_adam_optimizer
custom_optimizer_apply_gradients = kit_ops.custom_optimizer_apply_gradients


@ops.RegisterGradient("ReadEmbeddingVariableOp")
def _PluginReadEmbeddingVariableBprop(op, top_grad):
    return top_grad, None

@ops.RegisterGradient("PluginSparseFprop")
def _PluginSparseBackProp(op, top_grad):
    emb_var_grads_value, value_index = plugin_bprop(emb_handle=op.inputs[0], 
                                                    global_replica_id=op.inputs[4],
                                                    top_gradient=top_grad,
                                                    unique_op_name=op.get_attr("unique_op_name"))
    
    grads = ops.IndexedSlices(values=emb_var_grads_value,
                              indices=value_index)

    return [None] + [grads] + [None for _ in op.inputs[2:]]


@ops.RegisterGradient("PluginDenseFprop")
def _PluginDenseBackProp(op, top_grad):
    emb_var_grads_value, value_index = plugin_bprop(emb_handle=op.inputs[0], 
                                                    global_replica_id=op.inputs[3],
                                                    top_gradient=top_grad,
                                                    unique_op_name=op.get_attr("unique_op_name"))
    
    grads = ops.IndexedSlices(values=emb_var_grads_value,
                              indices=value_index)
                    
    return [None] + [grads] + [None for _ in op.inputs[2:]]
