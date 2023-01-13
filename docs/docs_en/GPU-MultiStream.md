# GPU Multi-stream

## Introduction

In training scenarios, GPUs are often used to accelerate computation. Since different computing kernels are only committed on the same stream, insufficient execution concurrency and low GPU utilization may occur under some models. To this end, we provide GPU Multi-stream optimization.

This feature provides multiple GPU streams, and has a variety of built-in graph splitting rules, and users can also manually specify the sub-graph. This feature enables several subgraphs without data dependency to be submitted to different GPU streams for execution, achieving concurrent execution at the subgraph level, thereby improving GPU utilization.

## User API

This feature can be enabled by setting the following parameters in `tf.ConfigProto`.

```python
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
sess_config = tf.ConfigProto()
sess_config.graph_options.rewrite_options.use_multi_stream = (rewriter_config_pb2.RewriterConfig.ON) # Turn on the multi-stream feature
sess_config.graph_options.rewrite_options.multi_stream_opts.multi_stream_num = 4 # The number of streams
```

## Graph Splitting Strategy

### 1. Manual Graph Splitting

For manual graph splitting, we provide the `tf.stream()` API to specify the `stream id`, which can be nested.

At the same time, the `tf.colocate_with()` API also supports the requirement of associating the newly created operation with one specified operation and placing them on the same stream.

#### Usage

```python
with tf.stream(0): 
  # Set the context of stream 0, and the stream id should be less than the number of streams specified in tf.ConfigProto
  a = tf.placeholder(tf.float32, [None, 1], name='a') # This operation will be placed on stream 0
  with tf.stream(1): 
    # Set the context of stream 1, and the stream id should be less than the number of streams specified in tf.ConfigProto
    b = tf.placeholder(tf.float32, [None, 1], name='b') # This operation will be placed on stream 1
  # Go back to the context of stream 0
  c = tf.constant([1, 2, 3, 4], tf.float32, [4, 1], name='c') # This operation will be placed on stream 0

with tf.colocate_with(a): 
  # Associated with `a`
  d = tf.constant([5, 6, 7, 8], tf.float32, [4, 1], name='d') # This operation is associated with `a`, and will be placed on the same GPU stream of `a`
```

#### Best Practise

```python
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

learning_rate = 0.01
max_train_steps = 1000
log_step = 100

train_X = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779],
                   [6.182], [7.59], [2.167], [7.042], [10.791], [5.313], [7.997],
                    [5.654], [9.27], [3.1]], dtype=np.float32)
train_Y = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779],
                   [6.182], [7.59], [2.167], [7.042], [10.791], [5.313], [7.997],
                    [5.654], [9.27], [3.1]], dtype=np.float32)
train_Z = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.336],
                    [2.596], [2.53], [1.221], [2.827], [3.465], [1.65], [2.904],
                    [2.42], [2.94], [1.3]], dtype=np.float32)

total_samples = train_X.shape[0]

Z_ = tf.placeholder(tf.float32, [None, 1])

with tf.stream(0):
    X = tf.placeholder(tf.float32, [None, 1])
    W_X = tf.Variable(tf.random_normal([1, 1]), name='weight_x')
    b = tf.Variable(tf.zeros([1]), name='bias')
    X_Result = tf.matmul(X, W_X)
    X_Result = tf.add(X_Result, b)

with tf.stream(1):
    Y = tf.placeholder(tf.float32, [None, 1])
    W_Y = tf.Variable(tf.random_normal([1, 1]), name='weight_y')
    Y_Result = tf.matmul(Y, W_Y)

Z = X_Result + Y_Result
loss = tf.reduce_sum(tf.pow(Z-Z_, 2)) / (total_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

sess_config = tf.ConfigProto()
sess_config.graph_options.rewrite_options.use_multi_stream = (rewriter_config_pb2.RewriterConfig.ON)
sess_config.graph_options.rewrite_options.multi_stream_opts.multi_stream_num = 2
with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    print("Start training:")
    for step in range(max_train_steps):
        sess.run(train_op, feed_dict={X: train_X, Y: train_Y, Z_: train_Z})
        if step % log_step == 0:
            c = sess.run(loss, feed_dict={X: train_X, Y: train_Y, Z_: train_Z})
            print("Step:%d, loss==%.4f, W_X==%.4f, b==%.4f, W_Y=%.4f" %
                  (step, c, sess.run(W_X), sess.run(b), sess.run(W_Y)))
    final_loss = sess.run(loss, feed_dict={X: train_X, Y: train_Y, Z_: train_Z})
    w_x, b, w_y= sess.run([W_X, b, W_Y])
    print("Step:%d, loss==%.4f, W_X==%.4f, b==%.4f, W_Y=%.4f" %
                  (max_train_steps, final_loss, w_x, b, w_y))
    print("Linear Regression Model: Z=%.4f*X + %.4f*Y + %.4f" % (w_x, w_y, b))
```

## Enabling GPU MPS

This optimization can adapt GPU MPS (Multi-Process Service). Users can enable GPU MPS by following these steps.

1. The host starts the GPU MPS.

   ```bash
   nvidia-cuda-mps-control -d
   ```

2. Docker launch configuration (if training inside the container)

   The `--ipc=host` option needs to be added so that the GPU MPS can be communicated with the process in the container. The following is an example.

   ```bash
   sudo docker run -itd --name <container_name> --ipc=host --gpus='"device=0"' <image_id> bash
   ```
   
   In this example, GPU 0 is bound to the created container, and one GPU is visible in the container. The GPU MPS can be used by directly executing GPU training tasks in the container.

