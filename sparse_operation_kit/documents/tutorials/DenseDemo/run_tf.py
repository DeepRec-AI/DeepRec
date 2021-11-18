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

import tensorflow as tf
from models import TfDenseDemo
import argparse
import sys
import time
import numpy as np
sys.path.append("../")
import utility
import nvtx

def main(args):
    dataset = utility.TFDataset(filename=args.data_filename, 
                                batchsize=args.global_batch_size,
                                as_sparse_tensor=False, 
                                repeat=1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model = TfDenseDemo(global_batch_size=args.global_batch_size,
                        vocabulary_size=args.vocabulary_size,
                        slot_num=args.slot_num,
                        nnz_per_slot=args.nnz_per_slot,
                        num_dense_layers=args.num_dense_layers,
                        embedding_vec_size=args.embedding_vec_size)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit = model(inputs, training=True)
            loss = loss_fn(labels, logit)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    time_arr = []
    for i, (inputs, labels) in enumerate(dataset):
        if args.stop_at_iter > 0 and i >= args.stop_at_iter:
            break

        rng = nvtx.start_range(message="Iteration_" + str(i), color="blue")
        start_time = time.time()
        loss = _train_step(inputs, labels)
        time_arr.append(time.time()-start_time)
        
        nvtx.end_range(rng)
        print("[INFO]: Iteration: {}, loss={}".format(i, loss))
    
    print("Average iteration time (except 1st iteration): ", np.mean(time_arr[1:]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run DNN model with tensorflow")

    parser.add_argument("--data_filename", type=str,
                        help="the filename of training datas",
                        required=True)
    parser.add_argument("--global_batch_size", type=int,
                        required=True)
    parser.add_argument("--vocabulary_size", type=int, required=True)
    parser.add_argument("--slot_num", type=int, required=True,
                        help="the number of feature fields.")
    parser.add_argument("--nnz_per_slot", type=int, required=True,
                        help="the number of keys in each slot")
    parser.add_argument("--num_dense_layers", type=int, required=True,
                        help="the number of fully connected layers in this DNN model.")
    parser.add_argument("--embedding_vec_size", type=int, required=True,
                        help="the dimension of embedding vectors")
    parser.add_argument("--stop_at_iter", type=int, required=False,
                        help="early stop the process if iteration reachs this setting.",
                        default=-1)

    args = parser.parse_args()

    main(args)
