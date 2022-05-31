from __future__ import print_function

import numpy as np
import tensorflow as tf
import datetime
import time
import threading

log_device_placement = True

# Num of multiplications to perform
n = 500

DIM = 1

THREAD_NUM = 4

# Create random large matrix
A = np.random.rand(DIM, DIM).astype('float32')
B = np.random.rand(DIM, DIM).astype('float32')

# Create a graph to store results
c1 = []
c2 = []

def dooo(M, n):
  if n < 1:
    return M
  else:
    return tf.matmul(M, dooo(M, n-1))

'''
Single GPU computing
'''
with tf.device('/gpu:0'):
  a = tf.placeholder(tf.float32, [DIM, DIM])
  b = tf.placeholder(tf.float32, [DIM, DIM])
  c1.append(dooo(a, n))
  c1.append(dooo(b, n))

  sum = tf.add_n(c1) #Addition of all elements in c1

# set multi_stream_num here
#
sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement,
                  multi_streams_num = 3))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
for i in range(300):
  sess.run(sum, {a:A, b:B})

def run(sess):
  for i in range(100):
    sess.run(sum, {a:A, b:B})

ths = []

for i in range(THREAD_NUM):
  ths.append(threading.Thread(target=run, args=(sess,)))

t1_1 = time.time()
t1_1 = int(round(t1_1 * 1000))

for i in range(THREAD_NUM):
  ths[i].start()

for i in range(THREAD_NUM):
  ths[i].join()

t2_1 = time.time()
t2_1 = int(round(t2_1 * 1000))

print ("Time cost: ", t2_1 - t1_1)

'''
Multi GPU computing
'''
'''
with tf.device('/gpu:0'):
  a = tf.placeholder(tf.float32, [DIM, DIM])
  c2.append(dooo(a, n))

with tf.device('/gpu:1'):
  b = tf.placeholder(tf.float32, [DIM, DIM])
  c2.append(dooo(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c2)

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
  sess.run(sum, {a:A, b:B})
t2_2 = datetime.datetime.now()
'''

