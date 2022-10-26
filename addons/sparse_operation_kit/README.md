# Embedding Variable SparseOperationKit Support
## Introduction
This doc introduces how to use [SparseOperationKit(SOK)](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/index.html) lookup together with [Embedding Variable](https://deeprec.readthedocs.io/zh/latest/Embedding-Variable.html#embedding-variable) in [Deeprec](https://github.com/alibaba/DeepRec). Users can now leverage both the highly efficient lookup operation provided in SOK and the flexible functionality with the DeepRec `Embedding Variable`. This doc includes 3 parts:
1. The API of SOK embedding lookup.
2. A demo that shows how to use SOK embedding lookup together with `Embedding Variable`, with just a few lines change.
3. A guide about how to build and test this new feature.
## API
```
def lookup_sparse(params,
                  indices,
                  hotness,
                  combiners)
```
* `params`: list of variables. Each variable should be created by `tf.get_embedding_variable`
* `indices`: list of tf.SparseTensor. The indices/keys to lookup.
* `hotness`: list of integer. The max hotness of each indices.
* `combiners`: list of string. The combiner type. Can be `mean` or `sum`

## User Example
```python
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd

from sparse_operation_kit import experiment as sok

# 1. init horovod and sok
hvd.init()
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
sok.init()

var = tf.get_embedding_variable("var_0",
                                embedding_dim=3,
                                initializer=tf.ones_initializer(tf.float32))
# 2. set target_gpu for Embedding Variable
var.target_gpu=-1

indices = tf.SparseTensor(
  indices=tf.convert_to_tensor([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]], dtype=tf.int64),
  values=tf.convert_to_tensor([1, 1, 3, 4, 5], dtype=tf.int64),
  dense_shape=[2, 3]
)
# 3. Use sok lookup_sparse
emb = sok.lookup_sparse([var], [indices], hotness=[3], combiners=['sum'])
fun = tf.multiply(emb, 2.0, name='multiply')
loss = tf.reduce_sum(fun, name='reduce_sum')
opt = tf.train.AdagradOptimizer(0.1)

g_v = opt.compute_gradients(loss)
train_op = opt.apply_gradients(g_v)
loss = hvd.allreduce(loss, op=hvd.Sum)

init = tf.global_variables_initializer()

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=sess_config) as sess:
  sess.run([init])
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))
  print(sess.run([emb, train_op, loss]))
```
You can also try the [demo.py](./example/demo.py) by using `horovodrun -np ${NUM_GPU} -H localhost:${NUM_GPU} python3 demo.py`.

There are 2 additional steps to use SOK embedding lookup with `Embedding Variable` compared with using `tf.nn.lookup`.
1. Initialize the horovod and SOK at the beginning of your script. SOK is using horovod as its communication backend. And SOK also needs explicit initialization to allocate necessary buffer/communicate necessary data before training.
2. Specify `target_gpu` for `Embedding Variable`. `target_gpu` is used to specify which GPU you want to place this variable on. There are 2 options supported right now:
  * -1: this means you want to distribute current variable into all GPUs
  * ${GPU_ID}: an integer that belongs to 0~${NUM_GPU} - 1, so you put this embedding variable to the specified GPU.
## Build and Run Utest
1. Clone Deeprec
```
git clone --recursive https://github.com/alibaba/DeepRec.git /DeepRec
```
2. Build DeepRec from source code. You can follow the instruction in [DeepRec Build](https://deeprec.readthedocs.io/zh/latest/DeepRec-Compile-And-Install.html).
3. Build and install SOK. You need to specify ${DeepRecBuild}, which is the directory that you store the DeepRec building intermidia results.
```
DeepRecBuild=${DeepRecBuild} cmake -DENABLE_DEEPREC=ON -DSM=75 .. && make -j && make install;
export PYTHONPATH=/DeepRec/addons/sparse_operation_kit/hugectr/sparse_operation_kit
```
4. Run utest.
```
cd /DeepRec/addons/sparse_operation_kit/adapter && horovodrun -np ${NUM_GPU} -H localhost:${NUM_GPU} python3 embedding_var_lookup_utest.py
```
## Benchmark
1. Download Kaggle Display Advertising Challenge Dataset (Criteo Dataset) from https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
2. Install Horovod using `HOROVOD_NCCL_LINK=SHARED HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod`
3. Run benchmark. 
```python
# 8 GPU
horovodrun -np 8 python benchmark_sok.py --batch_size 65536 
```