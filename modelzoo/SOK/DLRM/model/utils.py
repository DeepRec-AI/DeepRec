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
import sparse_operation_kit as sok
from model.models import SOKEmbedding
import os, glob



class WarmUpAndPolyDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate callable for the embeddings.
    Linear warmup on [0, warmup_steps] then
    Constant on [warmup_steps, decay_start_steps]
    And polynomial decay on [decay_start_steps, decay_start_steps + decay_steps].
    """

    def __init__(self,
                batch_size: int,
                decay_exp: float = 2.0,
                learning_rate: float = 40.0,
                warmup_steps: int = 8000,
                decay_steps: int = 12000,
                decay_start_steps: int = 10000):
        super(WarmUpAndPolyDecay, self).__init__()
        self.batch_size = batch_size
        self.decay_exp = decay_exp
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_start_steps = decay_start_steps

    def __call__(self, step):
        decay_exp = self.decay_exp
        learning_rate = self.learning_rate
        warmup_steps = self.warmup_steps
        decay_steps = self.decay_steps
        decay_start_steps = self.decay_start_steps

        scal = self.batch_size / 2048

        adj_lr = learning_rate * scal
        if warmup_steps == 0:
            return adj_lr

        warmup_lr = step / warmup_steps * adj_lr
        global_step = tf.cast(step, tf.float32)
        decay_steps = tf.cast(decay_steps, tf.float32)
        decay_start_step = tf.cast(decay_start_steps, tf.float32)
        warmup_lr = tf.cast(warmup_lr, tf.float32)

        steps_since_decay_start = global_step - decay_start_step
        already_decayed_steps = tf.minimum(steps_since_decay_start, decay_steps)
        decay_lr = adj_lr * (
            (decay_steps - already_decayed_steps) / decay_steps)**decay_exp
        decay_lr = tf.maximum(0.0001, decay_lr)

        lr = tf.where(
            global_step < warmup_steps, warmup_lr,
            tf.where(
                tf.logical_and(decay_steps > 0, global_step > decay_start_step),
                decay_lr, adj_lr))

        lr = tf.maximum(0.01, lr)
        return lr

    def get_config(self):
        return {
            'batch_size': self.batch_size,
            'decay_exp': self.decay_exp,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'decay_start_steps': self.decay_start_steps
        }


def get_optimizer(optimizer=None):
    if not optimizer:
        return sok.tf.keras.optimizers.Adam()
    else:
        return tf.keras.optimizers.get(optimizer)

def get_lr_callable(global_batch_size,
                    decay_exp,
                    learning_rate,
                    warmup_steps,
                    decay_steps,
                    decay_start_steps):
    return WarmUpAndPolyDecay(
        batch_size=global_batch_size,
        decay_exp=decay_exp,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_start_steps=decay_start_steps)


def shard_filenames(file_pattern, num_pipelines, pipeline_id):
    matching_files = glob.glob(file_pattern)
    matching_files.sort()

    nums_per_shard = len(matching_files) // num_pipelines
    return matching_files[pipeline_id * nums_per_shard : (pipeline_id + 1) * nums_per_shard]


def split_embedding_variables_from_others(model):
    if isinstance(model.embedding_layer, SOKEmbedding):
        return sok.split_embedding_variable_from_others(model.trainable_variables)
    else:
        dense_vars = []
        for layer in model.layers:
            if layer != model.embedding_layer:
                dense_vars.extend(layer.trainable_variables)
        return model.embedding_layer.trainable_variables, dense_vars


def apply_gradients(optimizer, variables, grads, using_sok):
    if using_sok:
        with sok.OptimizerScope(variables):
            opt = optimizer.apply_gradients(zip(grads, variables))
    else:
        opt = optimizer.apply_gradients(zip(grads, variables))
    return opt

