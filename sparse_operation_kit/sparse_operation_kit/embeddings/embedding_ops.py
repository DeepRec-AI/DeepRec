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

from tensorflow.distribute import MirroredStrategy, get_replica_context, has_strategy, get_strategy

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