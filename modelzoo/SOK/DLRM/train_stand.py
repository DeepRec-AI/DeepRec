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
import os
import argparse
import tensorflow as tf
import sys, os
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../../sparse_operation_kit/"
        )
    )
)
import sparse_operation_kit as sok
import model.utils as utils
from model.models import DLRM
import model.strategy_wrapper as strategy_wrapper
from model.dataset import BinaryDataset, BinaryDataset2
import time
import sys


def main(args):
    comm_options = None

    if args.distributed_tool == "onedevice":
        import horovod.tensorflow as hvd

        hvd.init()
        avaiable_cuda_devices = ",".join(
            [str(gpu_id) for gpu_id in range(args.gpu_num)]
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = avaiable_cuda_devices
        strategy = strategy_wrapper.OneDeviceStrategy()
        args.task_id = 0

    elif args.distributed_tool == "horovod":
        import horovod.tensorflow as hvd

        hvd.init()
        strategy = strategy_wrapper.HorovodStrategy()
        args.task_id = hvd.local_rank()
        args.gpu_num = hvd.size()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.task_id)
    else:
        raise ValueError(
            f"{args.distributed_tool} is not supported."
            f"Can only be one of {'onedevice',  'horovod'}"
        )

    with strategy.scope():
        if args.embedding_layer == "SOK":
            sok_init_op = sok.Init(global_batch_size=args.global_batch_size)

        model = DLRM(
            vocab_size=args.vocab_size_list,
            num_dense_features=args.num_dense_features,
            embedding_layer=args.embedding_layer,
            embedding_vec_size=args.embedding_vec_size,
            bottom_stack_units=args.bottom_stack,
            top_stack_units=args.top_stack,
            num_gpus = hvd.size(),
            comm_options=comm_options,
        )

        lr_callable = utils.get_lr_callable(
            global_batch_size=args.global_batch_size,
            decay_exp=args.decay_exp,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps,
            decay_start_steps=args.decay_start_steps,
        )

        embedding_optimizer = utils.get_optimizer(args.embedding_optimizer)
        embedding_optimizer.learning_rate = lr_callable
        dense_optimizer = utils.get_optimizer("SGD")
        dense_optimizer.learning_rate = lr_callable

    batch_size = (
        args.global_batch_size
        if args.distributed_tool == "onedevice"
        else args.global_batch_size // args.gpu_num
    )

    train_dataset = BinaryDataset2(
         os.path.join(args.train_file_pattern, "label.bin"),
         os.path.join(args.train_file_pattern, "dense.bin"),
         os.path.join(args.train_file_pattern, "category.bin"),
         batch_size= batch_size,
         drop_last=True,
         prefetch=10,
         global_rank=hvd.rank(),
         global_size=hvd.size(),
     )

    val_dataset = BinaryDataset(
        os.path.join(args.test_file_pattern, "label.bin"),
        os.path.join(args.test_file_pattern, "dense.bin"),
        os.path.join(args.test_file_pattern, "category.bin"),
        batch_size=batch_size,
        drop_last=True,
        prefetch=10,
        global_rank=hvd.rank(),
        global_size=hvd.size(),
    )

    

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")

    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(
            loss, global_batch_size=args.global_batch_size
        )

    def _train_step(dense, category, labels, first_batch=False):
        def _step_fn(dense, category, labels):
            logits = model(dense, category, training=True)
            loss = _replica_loss(labels, logits)
            emb_vars, other_vars = utils.split_embedding_variables_from_others(model)
            grads = tf.gradients(loss, emb_vars + other_vars, colocate_gradients_with_ops=True, unconnected_gradients=tf.UnconnectedGradients.NONE)
            emb_grads, other_grads = grads[: len(emb_vars)], grads[len(emb_vars) :]
            
            with tf.control_dependencies([*emb_grads]):
                emb_train_op = utils.apply_gradients(embedding_optimizer,emb_vars, emb_grads, args.embedding_layer == "SOK")
                if args.embedding_layer != "SOK":
                    emb_grads = strategy.reduce("sum", emb_grads)

            with tf.control_dependencies([*other_grads]):
                other_grads = strategy.reduce("sum", other_grads)
                other_train_op = utils.apply_gradients(dense_optimizer, other_vars, other_grads, False)
            if first_batch:
                strategy.broadcast_variables(other_vars)
                strategy.broadcast_variables(dense_optimizer.variables())

                if args.embedding_layer == "TF":
                    strategy.broadcast_variables(emb_vars)
                    strategy.broadcast_variables(embedding_optimizer.variables())

            with tf.control_dependencies([emb_train_op, other_train_op]):
                loss = tf.identity(loss)
                return loss
        return strategy.run(_step_fn, dense, category, labels)


    dense = tf.placeholder(tf.float32, shape = [batch_size, 13])
    category = tf.placeholder(tf.float32, shape = [batch_size, 26])
    labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
    total_loss_first = _train_step(dense, category, labels, True)
    total_loss = _train_step(dense, category, labels, False)
    
    probs = model(dense, category, training=False)
    auc,update_op = tf.metrics.auc(labels = labels, predictions = probs, num_thresholds=8000, curve='ROC')
    
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        if args.embedding_layer == "TF_EV":
            sess.run(tf.get_collection(tf.GraphKeys.EV_INIT_VAR_OPS))
            sess.run(tf.get_collection(tf.GraphKeys.EV_INIT_SLOT_OPS))
        if args.embedding_layer == "SOK":
            sess.run(sok_init_op)
        sess.run([init_op])

        t = time.time()
        run_time = 0
        iteration_time, dataload_time = [], []
        dataload_start = time.time()

        for step, (dense_, category_, labels_) in enumerate(train_dataset):
            iteration_start = time.time()
            dataload_time.append(time.time() - dataload_start)
            if step == 0:
                loss_v = sess.run([total_loss_first], feed_dict = {dense:dense_, category:category_, labels:labels_})
            else:
                loss_v = sess.run([total_loss], feed_dict = {dense:dense_, category:category_, labels:labels_})
            iteration_time.append(time.time() - iteration_start)
            
            if step > 0 and step % 100 == 0:
                print('Iteration:%d\tloss:%.6f\ttime:%.2fs\tAvg:%.2fms/iter\tdataload:%.2fms/iter'%(step, loss_v[0], time.time() - t, 
                                                                                                    1000*sum(iteration_time)/len(iteration_time),
                                                                                                    1000*sum(dataload_time)/len(dataload_time)))
                run_time += time.time() - t
                t = time.time()
                iteration_time = []
   
            if (step > 0 and step % 10000 == 0) or step == (len(train_dataset) - 1):
                eval_t = time.time()
                for step, (dense_, category_, labels_) in enumerate(val_dataset):
                    auc_value, _ = sess.run([auc,update_op], feed_dict = {dense:dense_, category:category_, labels:labels_})
                print('Evaluate in %dth iteration, time:%.2fs, AUC: %.6f'%(step, time.time() - eval_t, auc_value))
                t += (time.time() - eval_t)
            dataload_start = time.time()
            
        print('Training time: %.2fs'%(run_time + time.time() - t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--global_batch_size", type=int, required=True)
    parser.add_argument("--train_file_pattern", type=str, required=True)
    parser.add_argument("--test_file_pattern", type=str, required=True)
    parser.add_argument("--embedding_layer", type=str, choices=["TF", "SOK", "TF_EV"], required=True)
    parser.add_argument("--embedding_vec_size", type=int, required=True)
    parser.add_argument("--embedding_optimizer", type=str, required=False, default="SGD")
    parser.add_argument("--bottom_stack", type=int, nargs="+", required=True)
    parser.add_argument("--top_stack", type=int, nargs="+", required=True)
    parser.add_argument("--distributed_tool",type=str,choices=["onedevice", "horovod"],required=True,)
    parser.add_argument("--gpu_num", type=int, required=False, default=1)
    parser.add_argument("--decay_exp", type=int, required=False, default=2)
    parser.add_argument("--learning_rate", type=float, required=False, default=1.25)
    parser.add_argument("--warmup_steps", type=int, required=False, default=-1)
    parser.add_argument("--decay_steps", type=int, required=False, default=30000)
    parser.add_argument("--decay_start_steps", type=int, required=False, default=70000)
    args = parser.parse_args()

    args.vocab_size_list  = [
        39884406,   39043,      17289,      7420,       20263,     3,
        7120,       1543,       63,         38532951,   2953546,   403346,
        10,         2208,       11938,      155,        4,         976,
        14,         39979771,   25641295,   39664984,   585935,    12972,
        108,        36
    ]
    if args.distributed_tool == "onedevice":
        args.vocab_size_list = [int(num/8)+1 for num in args.vocab_size_list]
    args.num_dense_features = 13


    main(args)
