#	MultiStream

## 背景

在训练场景中，往往使用GPU对计算进行加速。由于不同的计算kernel只在同一个stream上提交，在某些模型下可能出现执行并发度不足，GPU利用率低的问题。为此，我们提供了GPU MultiStream 优化功能。

MultiStream用户功能提供了多个GPU Stream，并且内置了多种图划分规则，用户亦可手动指定分图。该功能可以使没有数据相关性的子图提交到不同GPU Stream上执行，实现子图级执行并发，从而提高GPU利用率。

## 用户接口

本功能在`tf.ConfigProto`中提供了如下接口用于开启MultiStream。

```python
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
sess_config = tf.ConfigProto()
sess_config.graph_options.rewrite_options.use_multi_stream = (rewriter_config_pb2.RewriterConfig.ON) # 开启multi-stream功能
sess_config.graph_options.rewrite_options.multi_stream_opts.multi_stream_num = 4 # 创建的stream数量
```

## 分图策略

### 1. 用户手动分图

针对手动分图，我们提供了`tf.stream()`编程接口，用于指定使用的`stream id`，该接口可嵌套。

同时，`tf.colocate_with()`接口也支持将新创建的op与指定op关联起来，放置到同一stream的需求。

#### 接口示例

```python
with tf.stream(0): 
  # 设定0号stream上下文，stream id要小于tf.ConfigProto中指定的创建的stream数量
  a = tf.placeholder(tf.float32, [None, 1], name='a') # 该op将放置到0号stream上
  with tf.stream(1): 
    # 设定1号stream上下文，stream id要小于tf.ConfigProto中指定的创建的stream数量
    b = tf.placeholder(tf.float32, [None, 1], name='b') # 该op将放置到1号stream上
  # 回到0号stream上下文
  c = tf.constant([1, 2, 3, 4], tf.float32, [4, 1], name='c') # 该op将放置到0号stream上

with tf.colocate_with(a): 
  # 与`a` op 关联
  d = tf.constant([5, 6, 7, 8], tf.float32, [4, 1], name='d') # 该op与`a` op相关联，即放置到同一个GPU stream上
```

#### 最佳实践

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

## 开启GPU MPS

本优化中依赖GPU MPS（Multi-Process Service) 优化。需要按以下步骤开启GPU MPS。

1. 宿主机 GPU MPS后台服务进程启动

   ```bash
   nvidia-cuda-mps-control -d
   ```

2. docker启动配置（如果需要在容器内训练）

   需要增加`--ipc=host`选项，使得容器内可以与GPU MPS后台进程进行通信。如下是个例子。

   ```bash
   sudo docker run -itd --name <container_name> --ipc=host --gpus='"device=0"' <image_id> bash
   ```

   本例中将0号GPU绑定到创建的容器上，容器内可见一个GPU。容器内直接执行GPU训练任务代码即可使用GPU MPS服务。

