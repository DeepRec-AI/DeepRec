# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for tensorflow.ops.tf.MSBatchMatMulGrad"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os
import shutil
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'

from tensorflow.contrib import layers
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.training import adagrad
from tensorflow.python.ops import array_ops



# run without auto-replacement of fused ops
def runNonFuse():
    g1 = ops.Graph()
    with g1.as_default():
        random_seed.set_random_seed(0)

        n_num = 1024
        q_num = 50
        k_num = 50
        c_num = 128  # c_num % split_num == 0
        split_num = 8

        data_float32_q = array_ops.placeholder(
                dtypes.float32, shape=(None, q_num, c_num))
        data_float32_k = array_ops.placeholder(
                dtypes.float32, shape=(None, k_num, c_num))

        x_float32 = data_float32_q
        y_float32 = data_float32_k
        m = variable_scope.get_variable(
                "m_non_fuse", [split_num, n_num, q_num, k_num],
                dtype=dtypes.int32,
                initializer=init_ops.random_uniform_initializer(0, 2))
        m_bool = math_ops.cast(m, dtype=dtypes.bool)
        m_bool = array_ops.reshape(m_bool, [-1, q_num, k_num])
        p_float32 = constant_op.constant(
                0, shape=[split_num*n_num, q_num, k_num],
                dtype=dtypes.float32)

        with ops.name_scope('NonFuseForward') as scope:
            with ops.device("/cpu:0"):

                x_float32 = layers.fully_connected(
                        x_float32, c_num,
                        activation_fn=nn_ops.leaky_relu, scope="X")

                y_float32 = layers.fully_connected(
                        y_float32, c_num,
                        activation_fn=nn_ops.leaky_relu, scope="Y")

                xs_float32 = array_ops.concat(
                        array_ops.split(x_float32, split_num, axis=2), axis=0)
                ys_float32 = array_ops.concat(
                        array_ops.split(y_float32, split_num, axis=2), axis=0)
                output_non_fuse_float32 = math_ops.matmul(
                        xs_float32, ys_float32,
                        transpose_a=False, transpose_b=True)

                zero_tensor = array_ops.zeros_like(array_ops.identity(output_non_fuse_float32))
                output_non_fuse_float32 = array_ops.where(
                        m_bool, output_non_fuse_float32, zero_tensor)
                zero_tensor2 =array_ops.zeros_like(zero_tensor)

                layer1_non_fuse_float32 = layers.fully_connected(
                        output_non_fuse_float32, 40,
                        activation_fn=nn_ops.leaky_relu)
                layer2_non_fuse_float32 = layers.fully_connected(
                        layer1_non_fuse_float32, 20,
                        activation_fn=nn_ops.leaky_relu)
                layer2_non_fuse_float32 = array_ops.reshape(
                        layer2_non_fuse_float32, [n_num, -1])
                layer3_non_fuse_float32 = layers.fully_connected(
                        layer2_non_fuse_float32, 1,
                        activation_fn=nn_ops.leaky_relu)
                labels_non_fuse_float32 = constant_op.constant(
                        1, shape=[n_num, 1], dtype=dtypes.float32)
                loss_op_non_fuse_float32 = math_ops.reduce_mean(
                        nn_impl.sigmoid_cross_entropy_with_logits(
                            logits=layer3_non_fuse_float32,
                            labels=labels_non_fuse_float32))

        with ops.name_scope('NonFuseBackward') as scope:
            with ops.device("/cpu:0"):
                train_op_non_fuse_float32 = adagrad.AdagradOptimizer(
                        learning_rate=0.0001,
                        initial_accumulator_value=0.1).minimize(
                                loss_op_non_fuse_float32)

        init_global = variables.global_variables_initializer()
        init_local = variables.local_variables_initializer()

        # trigger fusion op or not
        graph_options = config_pb2.GraphOptions(
                optimizer_options=config_pb2.OptimizerOptions(
                    do_op_fusion=False))
        config = config_pb2.ConfigProto(
                allow_soft_placement=False, graph_options=graph_options)
        with session.Session(config=config) as sess:
            from tensorflow.python.framework import graph_io
            graph_io.write_graph(sess.graph, './', 'train.pbtxt')

            # output the graph_def
            np.random.seed(0)
            feed_data_q = np.random.rand(n_num, q_num, c_num)
            feed_data_k = np.random.rand(n_num, k_num, c_num)

            sess.run([init_global, init_local])
            for step in range(50):
                loss_val_non_fuse, train_op_val = sess.run(
                        [loss_op_non_fuse_float32,
                         train_op_non_fuse_float32],
                        feed_dict={data_float32_q: feed_data_q,
                                   data_float32_k: feed_data_k})

            print("loss val non-fuse: %2.7f" % (loss_val_non_fuse))
            return loss_val_non_fuse


def runFuse():

    g2 = ops.Graph()
    with g2.as_default():
        random_seed.set_random_seed(0)

        n_num = 1024
        q_num = 50
        k_num = 50
        c_num = 128  # c_num % split_num == 0
        split_num = 8

        data_float32_q = array_ops.placeholder(
                dtypes.float32, shape=(None, q_num, c_num))
        data_float32_k = array_ops.placeholder(
                dtypes.float32, shape=(None, k_num, c_num))

        x_float32 = data_float32_q
        y_float32 = data_float32_k
        m = variable_scope.get_variable(
                "m_fuse", [split_num, n_num, q_num, k_num], dtype=dtypes.int32,
                initializer=init_ops.random_uniform_initializer(0, 2))
        m_bool = math_ops.cast(m, dtype=dtypes.bool)
        m_bool = array_ops.reshape(m_bool, [-1, q_num, k_num])
        p_float32 = constant_op.constant(
                0, shape=[split_num*n_num, q_num, k_num], dtype=dtypes.float32)

        with ops.name_scope('FuseForward') as scope:
            with ops.device("/cpu:0"):

                x_float32 = layers.fully_connected(
                        x_float32, c_num,
                        activation_fn=nn_ops.leaky_relu, scope="X")

                y_float32 = layers.fully_connected(
                        y_float32, c_num,
                        activation_fn=nn_ops.leaky_relu, scope="Y")

                xs_float32 = array_ops.concat(
                        array_ops.split(x_float32, split_num, axis=2), axis=0)
                ys_float32 = array_ops.concat(
                        array_ops.split(y_float32, split_num, axis=2), axis=0)
                output_fuse_float32 = math_ops.matmul(
                        xs_float32, ys_float32,
                        transpose_a=False, transpose_b=True)

                zero_tensor = array_ops.zeros_like(array_ops.identity(output_fuse_float32))
                output_fuse_float32 = array_ops.where(
                        m_bool, output_fuse_float32, zero_tensor)
                zero_tensor2 = array_ops.zeros_like(zero_tensor)

                layer1_fuse_float32 = layers.fully_connected(
                        output_fuse_float32, 40,
                        activation_fn=nn_ops.leaky_relu)
                layer2_fuse_float32 = layers.fully_connected(
                        layer1_fuse_float32, 20,
                        activation_fn=nn_ops.leaky_relu)
                layer2_fuse_float32 = array_ops.reshape(
                        layer2_fuse_float32, [n_num, -1])
                layer3_fuse_float32 = layers.fully_connected(
                        layer2_fuse_float32, 1,
                        activation_fn=nn_ops.leaky_relu)
                labels_fuse_float32 = constant_op.constant(
                        1, shape=[n_num, 1], dtype=dtypes.float32)
                loss_op_fuse_float32 = math_ops.reduce_mean(
                        nn_impl.sigmoid_cross_entropy_with_logits(
                            logits=layer3_fuse_float32,
                            labels=labels_fuse_float32))

        with ops.name_scope('FuseBackward') as scope:
            with ops.device("/cpu:0"):
                train_op_fuse_float32 = adagrad.AdagradOptimizer(
                        learning_rate=0.0001,
                        initial_accumulator_value=0.1).minimize(
                                loss_op_fuse_float32)

        init_global = variables.global_variables_initializer()
        init_local = variables.local_variables_initializer()

        # trigger fusion op or not
        graph_options = config_pb2.GraphOptions(
                optimizer_options=config_pb2.OptimizerOptions(
                    do_op_fusion=True))
        config = config_pb2.ConfigProto(
                allow_soft_placement=False, graph_options=graph_options)
        with session.Session(config=config) as sess:
            from tensorflow.python.framework import graph_io
            graph_io.write_graph(sess.graph, './', 'train2.pbtxt')

            np.random.seed(0)
            feed_data_q = np.random.rand(n_num, q_num, c_num)
            feed_data_k = np.random.rand(n_num, k_num, c_num)
            sess.run([init_global, init_local])

            for step in range(50):
                loss_val_replaced, train_op_val = sess.run(
                        [loss_op_fuse_float32, train_op_fuse_float32],
                        feed_dict={data_float32_q: feed_data_q,
                                   data_float32_k: feed_data_k})

            print("loss val fuse: %2.7f" % loss_val_replaced)
            return loss_val_replaced

def runFuseForIntType():
    graph_options = config_pb2.GraphOptions(
            optimizer_options=config_pb2.OptimizerOptions(
                do_op_fusion=True))
    config = config_pb2.ConfigProto(
            allow_soft_placement=False, graph_options=graph_options)

    with session.Session(config=config) as sess:
        with sess.graph.as_default():

            # with ops.name_scope('FuseForward') as scope:
            t_cond = variables.Variable([[True, True], [False, False]], dtype=dtypes.bool)
            t_then = variables.Variable([[11,12],[13,14]], dtype=dtypes.int32)
            t_else = variables.Variable([[21,22],[23,24]], dtype=dtypes.int32)
            t_out  = variables.Variable([[31,32],[33,34]], dtype=dtypes.int32)
            
            t_then = array_ops.zeros_like(array_ops.reshape(array_ops.unique(array_ops.reshape(t_then, [-1]))[0], [-1, 2]))
            t_select = array_ops.where(
                    t_cond, t_then, t_else)
            t_result = t_out + t_select

            init_global = variables.global_variables_initializer()
            init_local = variables.local_variables_initializer()

            from tensorflow.python.framework import graph_io
            graph_io.write_graph(sess.graph, './', 'train_3.pbtxt')

            np.random.seed(0)
            feed_p_input = np.random.rand(2, 2)
            sess.run([init_global, init_local])

            result = sess.run([t_result, ])
            print("result:", result)
            return result

class SelectZeroLikeFusionTest(test.TestCase):
    def testFusion(self):
        res_non_fuse = runNonFuse()
        res_fuse = runFuse()
        self.assertAllCloseAccordingToType(res_non_fuse, res_fuse)

    def testFusionForIntType(self):
        result = runFuseForIntType()
        self.assertAllEqual(result, [[[31, 32], [56, 58]]])

if __name__ == "__main__":
    test.main()
