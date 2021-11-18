# Known Issues #
There are several issues in SparseOperationKit, and we are trying to fix shose issues in the near future.

## Only TensorFlow 2.x is supported ##
Currently, only TensorFlow 2.x is supported.

## NCCL conflicts ##
In SparseOperationKit's Embedding Layers, NCCL is used to transfer data among different GPUs, while the synchronized training tool might also use NCCL to do the data transferring among GPUs, such as `allreduce` in `tf.distribute.Strategy` and `horovod`. In some cases, the synchronized training tool will call NCCL while SOK is also calling NCCL, which leads to multiple NCCL kernel racing and program hanging. 

The solution for such problem is to make the program does not call multiple NCCL APIs at the same time. For example, in TensorFlow, you can add `tf.control_dependencies()` between different NCCL APIs to order those calling. 