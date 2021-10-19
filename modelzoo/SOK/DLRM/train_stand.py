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
sys.path.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../")))
import sparse_operation_kit.sparse_operation_kit as sok
import model.utils as utils
from model.models import DLRM
import model.strategy_wrapper as strategy_wrapper
from model.dataset import CriteoTsvReader
import time


def main(args):
    comm_options = None

    if args.distributed_tool == "onedevice":
        avaiable_cuda_devices = ",".join([str(gpu_id) for gpu_id in range(args.gpu_num)])
        os.environ["CUDA_VISIBLE_DEVICES"] = avaiable_cuda_devices
        import horovod.tensorflow as hvd
        hvd.init()
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
        raise ValueError(f"{args.distributed_tool} is not supported.")


    with strategy.scope():
        if args.embedding_layer == "SOK":
            sok_init_op = sok.Init(global_batch_size=args.global_batch_size)

        model = DLRM(vocab_size=args.vocab_size_list,
                     num_dense_features=args.num_dense_features,
                     embedding_layer=args.embedding_layer,
                     embedding_vec_size=args.embedding_vec_size,
                     bottom_stack_units=args.bottom_stack,
                     top_stack_units=args.top_stack,
                     comm_options=comm_options)

        lr_callable = utils.get_lr_callable(global_batch_size=args.global_batch_size,
                                            decay_exp=args.decay_exp,
                                            learning_rate=args.learning_rate,
                                            warmup_steps=args.warmup_steps,
                                            decay_steps=args.decay_steps,
                                            decay_start_steps=args.decay_start_steps)

        embedding_optimizer = utils.get_optimizer(args.embedding_optimizer)
        embedding_optimizer.learning_rate = lr_callable
        dense_optimizer = utils.get_optimizer("Adam")

    batch_size = args.global_batch_size if args.distributed_tool == "onedevice" \
                                        else args.global_batch_size // args.gpu_num
    if args.distributed_tool != "onedevice":
        args.train_file_pattern = utils.shard_filenames(args.train_file_pattern, 
                                                        args.gpu_num, args.task_id)
        args.test_file_pattern = utils.shard_filenames(args.test_file_pattern,
                                                        args.gpu_num, args.task_id)

    train_dataset = CriteoTsvReader(file_pattern=args.train_file_pattern,
                                    num_dense_features=args.num_dense_features,
                                    vocab_sizes=args.vocab_size_list,
                                    batch_size=batch_size)()
    val_dataset = CriteoTsvReader(file_pattern=args.test_file_pattern,
                                  num_dense_features=args.num_dense_features,
                                  vocab_sizes=args.vocab_size_list,
                                  batch_size=batch_size)()
                                  
    train_iterator = train_dataset.make_initializable_iterator()
    iterator_init = train_iterator.initializer
    val_iterator = val_dataset.make_initializable_iterator()
    val_iterator_init = val_iterator.initializer


    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)


    def _train_step(features, labels, first_batch=False):
        def _step_fn(features, labels):
            logits = model(features, training=True)
            loss = _replica_loss(labels, logits) 
            emb_vars, other_vars = utils.split_embedding_variables_from_others(model)
            grads = tf.gradients(loss, emb_vars + other_vars, colocate_gradients_with_ops=True,
                                unconnected_gradients=tf.UnconnectedGradients.NONE)
            emb_grads, other_grads = grads[:len(emb_vars)], grads[len(emb_vars):]
            emb_train_op = utils.apply_gradients(embedding_optimizer, emb_vars, emb_grads, 
                            args.embedding_layer == "SOK", 
                            aggregate_gradients = True)

            with tf.control_dependencies([*emb_grads]):
                other_grads = strategy.reduce("sum", other_grads)
            other_train_op = utils.apply_gradients(dense_optimizer, other_vars, other_grads,
                            False)
                
            if first_batch:
                strategy.broadcast_variables(other_vars)
                strategy.broadcast_variables(dense_optimizer.variables())

                if args.embedding_layer == "TF":
                    strategy.broadcast_variables(emb_vars)
                    strategy.broadcast_variables(embedding_optimizer.variables())
            with tf.control_dependencies([emb_train_op, other_train_op]):
                total_loss = strategy.reduce("sum", loss)
                total_loss = tf.identity(total_loss)
                return total_loss
        return strategy.run(_step_fn, features, labels)


    def _val_step(features, labels):
        def _step_fn(features, labels):
            val_logits = model(features, training=False)
            val_loss = _replica_loss(labels, val_logits)
            val_loss = strategy.reduce("sum", val_loss)

            labels = tf.identity(labels)
            val_logits = strategy.gather(val_logits)
            labels = strategy.gather(labels)
            return val_logits, labels, val_loss
        return strategy.run(_step_fn, features, labels)


    features, labels = train_iterator.get_next()
    total_loss_first =  _train_step(features, labels, True)
    total_loss =  _train_step(features, labels)
    val_features, val_labels = val_iterator.get_next()
    val_logits, val_labels, val_loss = _val_step(val_features, val_labels)
    acc, _ = tf.compat.v1.metrics.accuracy(labels=val_labels, predictions=val_logits)
    auc, _ = tf.compat.v1.metrics.auc(labels=val_labels, predictions=val_logits, num_thresholds=1000)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.log_device_placement = False
    start_time = time.time()
    begin_time = start_time
    with tf.Session(config=config) as sess:
        if args.embedding_layer == "SOK":
            sess.run(sok_init_op)
        sess.run([init_op, iterator_init])
        sess.graph.finalize()

        for step in range(args.train_steps):
            try:
                if step == 0:
                    loss_v = sess.run([total_loss_first])
                else:
                    loss_v = sess.run([total_loss])
                if (step % 10 == 0):
                    print("Training complate:[{}/{}]".format(step, args.train_steps),f"Step: {step}, loss: {loss_v}")
                if (step == args.train_steps):
                    print("Training complate:[{}/{}]".format(step, args.train_steps))
            except tf.errors.OutOfRangeError:
                sess.run([iterator_init])

        end_time = time.time()
        if args.task_id == 0:
            print(f"With {args.distributed_tool} + {args.embedding_layer} embedding layer, "
                f"on {args.gpu_num} GPUs, and global_batch_size is {args.global_batch_size}, "
                f"it takes {end_time - start_time} seconds to "
                f"finish {args.train_steps} steps training for DLRM.")

        
        sess.run([val_iterator_init])
        for step in range(1, args.test_steps + 1):
            try:
                v_logits, v_labels, v_loss = sess.run([val_logits, val_labels, val_loss])
                acc_v, auc_v = sess.run([acc, auc])
                if (step % 10 == 0):
                    print("Evaluation complate:[{}/{}]".format(step, args.test_steps))
                if (step == args.test_steps):
                    print("Evaluation complate:[{}/{}]".format(step, args.test_steps))
                    print("ACC = {}\nAUC = {}".format(acc_v, auc_v))
            except tf.errors.OutOfRangeError:
                sess.run([val_iterator_init])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--global_batch_size", type=int, required=True)
    parser.add_argument("--train_file_pattern", type=str, required=True)
    parser.add_argument("--test_file_pattern", type=str, required=True)
    parser.add_argument("--embedding_layer", type=str, choices=["TF", "SOK"], required=True)
    parser.add_argument("--embedding_vec_size", type=int, required=True)
    parser.add_argument("--embedding_optimizer", type=str, required=False, default='SGD')
    parser.add_argument("--bottom_stack", type=int, nargs="+", required=True)
    parser.add_argument("--top_stack", type=int, nargs="+", required=True)
    parser.add_argument("--distributed_tool", type=str, 
                        choices=["onedevice", "horovod"],
                        required=True)
    parser.add_argument("--gpu_num", type=int, required=False, default=1)
    parser.add_argument("--decay_exp", type=int, required=False, default=2)
    parser.add_argument("--learning_rate", type=float, required=False, default=1.25)
    parser.add_argument("--warmup_steps", type=int, required=False, default=8000)
    parser.add_argument("--decay_steps", type=int, required=False, default=30000)
    parser.add_argument("--decay_start_steps", type=int, required=False, default=70000)
    parser.add_argument("--test_steps", type=int, required=False, default=100)
    parser.add_argument("--train_steps", type=int, required=False, default=100)

    args = parser.parse_args()

    args.vocab_size_list = [39884407, 39043, 17289, 7420, 20263, 
                            3, 7120, 1543, 63, 38532952, 2953546, 
                            403346, 10, 2208, 11938, 155, 4, 976, 
                            14, 39979772, 25641295, 39664985, 585935, 
                            12972, 108, 36]
    args.num_dense_features = 13
    args.train_steps = 4195155968 // args.global_batch_size if args.train_steps == -1 else args.train_steps


    if (args.distributed_tool == "onedevice" and args.gpu_num != 1):
        raise ValueError(f"When 'onedevice' is used as the distributed_tool, "
                         f"gpu_num must be 1, which is {args.gpu_num}")
    elif(args.distributed_tool == "horovod"):
        # gpu_num will be ignored.
        rank_size = os.getenv("OMPI_COMM_WORLD_SIZE")
        if rank_size is None:
            raise ValueError(f"When distributed_tool is set to {args.distributed_tool}, "
                             "mpiexec / mpirun must be used to launch this program.")


    main(args)
    
