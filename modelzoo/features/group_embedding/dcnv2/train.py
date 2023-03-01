# Copyright (c) 2022 Intel Corporation

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#    http://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

# ==============================================================================

import numpy as np

from ast import arg

import time

import argparse

import os

import sys

import math

import json

import collections

import tensorflow as tf

from tensorflow.python.client import timeline

from tensorflow.python.framework import dtypes

from tensorflow.python.framework import sparse_tensor

from tensorflow.python.feature_column import feature_column_v2 as fc

from tensorflow.python.ops import partitioned_variables

from tensorflow.python.framework import ops

os.environ["TF_GPU_THREAD_MODE"] = "global"

import horovod.tensorflow as hvd

#Enable group_embedding_lookup
tf.config.experimental.enable_distributed_strategy(strategy="collective")

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants

CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive

CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive

LABEL_COLUMN = ['clicked']

TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

HASH_BUCKET_SIZES = {
    'C1': 2500,
    'C2': 2000,
    'C3': 300000,
    'C4': 250000,
    'C5': 1000,
    'C6': 100,
    'C7': 20000,
    'C8': 4000,
    'C9': 20,
    'C10': 100000,
    'C11': 10000,
    'C12': 250000,
    'C13': 40000,
    'C14': 100,
    'C15': 100,
    'C16': 200000,
    'C17': 50,
    'C18': 10000,
    'C19': 4000,
    'C20': 20,
    'C21': 250000,
    'C22': 100,
    'C23': 100,
    'C24': 250000,
    'C25': 400,
    'C26': 100000
}

EMBEDDING_DIMENSIONS = {
    'C1': 64,
    'C2': 64,
    'C3': 128,
    'C4': 128,
    'C5': 64,
    'C6': 64,
    'C7': 64,
    'C8': 64,
    'C9': 64,
    'C10': 128,
    'C11': 64,
    'C12': 128,
    'C13': 64,
    'C14': 64,
    'C15': 64,
    'C16': 128,
    'C17': 64,
    'C18': 64,
    'C19': 64,
    'C20': 64,
    'C21': 128,
    'C22': 64,
    'C23': 64,
    'C24': 128,
    'C25': 64,
    'C26': 128
}


def transform_numeric(feature):
    r'''Transform numeric features.

    '''

    # Notes: Statistics of Kaggle's Criteo Dataset has been calculated in advance to save time.

    mins_list = [
        0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    range_list = [
        1539.0, 22069.0, 65535.0, 561.0, 2655388.0, 233523.0, 26297.0, 5106.0,
        24376.0, 9.0, 181.0, 1807.0, 6879.0
    ]

    def make_minmaxscaler(min, range):

        def minmaxscaler(col):

            return (col - min) / range

        return minmaxscaler

    numeric_list = []

    for column_name in CONTINUOUS_COLUMNS:

        normalizer_fn = None

        i = CONTINUOUS_COLUMNS.index(column_name)

        normalizer_fn = make_minmaxscaler(mins_list[i], range_list[i])

        numeric = normalizer_fn(feature[i])

        numeric_list.append(tf.reshape(numeric, shape=[-1, 1]))

    return numeric_list

def transform_feature_column():

    feature_columns = []

    mins_list = [
        0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    range_list = [
        1539.0, 22069.0, 65535.0, 561.0, 2655388.0, 233523.0, 26297.0, 5106.0,
        24376.0, 9.0, 181.0, 1807.0, 6879.0
    ]

    def make_minmaxscaler(min, range):

        def minmaxscaler(col):

            return (col - min) / range

        return minmaxscaler


    for column_name in CONTINUOUS_COLUMNS:

        normalizer_fn = None

        i = CONTINUOUS_COLUMNS.index(column_name)

        normalizer_fn = make_minmaxscaler(mins_list[i], range_list[i])

        column = tf.feature_column.numeric_column(column_name,
                                                  normalizer_fn=normalizer_fn)

        feature_columns.append(column)

    with tf.feature_column.group_embedding_column_scope(name="categorical"):
        for i, column_name in enumerate(CATEGORICAL_COLUMNS):

            ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                                filter_option=None)

            column = tf.feature_column.categorical_column_with_embedding(
                key = column_name,
                dtype=tf.int64,
                ev_option = ev_opt
            )
            
            with tf.device("/gpu:0"):
                weight = tf.feature_column.embedding_column(
                    categorical_column=column,
                    dimension=EMBEDDING_DIMENSIONS[column_name],
                    initializer=tf.ones_initializer(tf.float32),
                )

            feature_columns.append(weight)

    return feature_columns

def transform_features(sparse_features, dense_features):
    features = {}

    for column_name in CONTINUOUS_COLUMNS:

        i = CONTINUOUS_COLUMNS.index(column_name)

        numeric = dense_features[i]

        features[column_name] = numeric
    
    max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max

    for i, column_name in enumerate(CATEGORICAL_COLUMNS):
        category = tf.strings.to_hash_bucket_fast(sparse_features[i], max_value)
        # ragged_tensor = tf.RaggedTensor.from_row_lengths(
        #        values=category, row_lengths=tf.ones_like(category))

        sparse_tensor = fc._to_sparse_input_and_drop_ignore_values(
                category)

        sparse_tensor = tf.sparse.reshape(sparse_tensor, (-1, 1))

        features[column_name] = sparse_tensor
    
    return features

def transform_categorical(feature):

    max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max

    variables = []

    indices = []

    for i, column_name in enumerate(CATEGORICAL_COLUMNS):

        ev_opt = tf.EmbeddingVariableOption(evict_option=None,
                                            filter_option=None)

        with tf.device('/gpu:{}'.format(0)):
            embedding_weights = tf.get_embedding_variable(
                f'{column_name}_weight',
                initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.05
                ),
                embedding_dim=EMBEDDING_DIMENSIONS[column_name],
                ev_option=ev_opt)
            

        category = tf.strings.to_hash_bucket_fast(feature[i], max_value)


        i = CATEGORICAL_COLUMNS.index(column_name)

        target_gpu = i % hvd.size()

        target_gpu = -1

        embedding_weights.target_gpu = target_gpu


        ragged_tensor = tf.RaggedTensor.from_row_lengths(
            values=category, row_lengths=tf.ones_like(category))


        indices.append(ragged_tensor)

        variables.append(embedding_weights)


    combiners = ['sum' for _ in range(len(CATEGORICAL_COLUMNS))]

    deep_features = tf.nn.group_embedding_lookup_sparse(variables, indices, combiners)


    return deep_features


def stacked_dcn_v2(features, mlp_dims):
    r'''Stacked DCNv2.

    DCNv2: Improved Deep & Cross Network and Practical Lessons for Web-scale

    Learning to Rank Systems.

    See https://arxiv.org/abs/2008.13535 for more information.

    '''

    with tf.name_scope('cross'):
        if args.use_feature_columns:

            cross_input = features
        else:
            cross_input = tf.concat(features, axis=-1)

            cross_input_shape = [-1, sum([f.shape[-1] for f in features])]

            cross_input = tf.reshape(cross_input, cross_input_shape)
        
        cross_input_sq = tf.layers.dense(
            cross_input,
            cross_input.shape[-1],
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(),
            bias_initializer=tf.zeros_initializer())

        cross_output = cross_input * cross_input_sq + cross_input

        cross_output = tf.reshape(cross_output, [-1, cross_input.shape[1]])

        
        if args.use_feature_columns:
            cross_output_dim = (len(CATEGORICAL_COLUMNS+CONTINUOUS_COLUMNS) * (len(CATEGORICAL_COLUMNS+CONTINUOUS_COLUMNS) + 1)) / 2
        else:
            cross_output_dim = (len(features) * (len(features) + 1)) / 2

    with tf.name_scope('mlp'):

        prev_layer = cross_output

        prev_dim = cross_output_dim

        for i, d in enumerate(mlp_dims[:-1]):

            prev_layer = tf.layers.dense(
                prev_layer,
                d,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=math.sqrt(2.0 / (prev_dim + d))),
                bias_initializer=tf.random_normal_initializer(mean=0.0,
                                                              stddev=math.sqrt(
                                                                  1.0 / d)),
                name=f'mlp_{i}')

            prev_dim = d

        return tf.layers.dense(prev_layer,
                               mlp_dims[-1],
                               activation=tf.nn.sigmoid,
                               kernel_initializer=tf.random_normal_initializer(
                                   mean=0.0,
                                   stddev=math.sqrt(
                                       2.0 / (prev_dim + mlp_dims[-1]))),
                               bias_initializer=tf.random_normal_initializer(
                                   mean=0.0,
                                   stddev=math.sqrt(1.0 / mlp_dims[-1])),
                               name=f'mlp_{len(mlp_dims) - 1}')


# generate dataset pipline


def build_model_input(filename, batch_size, num_epochs):

    def parse_csv(value):

        tf.logging.info('Parsing {}'.format(filename))

        cont_defaults = [[0.0] for i in range(1, 14)]

        cate_defaults = [[' '] for i in range(1, 27)]

        label_defaults = [[0]]

        column_headers = TRAIN_DATA_COLUMNS

        record_defaults = label_defaults + cont_defaults + cate_defaults

        columns = tf.io.decode_csv(value, record_defaults=record_defaults)

        all_columns = collections.OrderedDict(zip(column_headers, columns))

        labels = all_columns.pop(LABEL_COLUMN[0])

        dense_feature = [all_columns[name] for name in CONTINUOUS_COLUMNS]

        sparse_feature = [all_columns[name] for name in CATEGORICAL_COLUMNS]

        return dense_feature, sparse_feature, labels

    '''Work Queue Feature'''

    if args.workqueue and not args.tf:

        from tensorflow.python.ops.work_queue import WorkQueue

        work_queue = WorkQueue([filename])

        # For multiple files：

        # work_queue = WorkQueue([filename, filename1,filename2,filename3])

        files = work_queue.input_dataset()

    else:

        files = filename

    # Extract lines from input files using the Dataset API.

    dataset = tf.data.TextLineDataset(files)

    dataset = dataset.shuffle(buffer_size=20000,
                              seed=args.seed)  # fix seed for reproducing

    dataset = dataset.repeat(num_epochs)

    dataset = dataset.batch(batch_size)

    dataset = dataset.map(parse_csv, num_parallel_calls=28)

    dataset = dataset.prefetch(2)

    return dataset


def main():

    # check dataset and count data set size

    print("Checking dataset...")

    train_file = args.data_location + '/train.csv'

    if (not os.path.exists(train_file)):

        print("Dataset does not exist in the given data_location.")

        sys.exit()

    no_of_training_examples = sum(1 for line in open(train_file))

    print("Numbers of training dataset is {}".format(no_of_training_examples))

    # set batch size, eporch & steps

    assert args.batch_size % hvd.size() == 0

    batch_size = int(args.batch_size / hvd.size())

    if args.steps == 0:

        no_of_epochs = 1

        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)

    else:

        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)

        train_steps = args.steps

    print("The training steps is {}".format(train_steps))

    # set fixed random seed

    tf.set_random_seed(args.seed)

    # create data pipline of train & test dataset

    with tf.device('/cpu:0'):

        train_dataset = build_model_input(train_file, batch_size, no_of_epochs)

        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

        next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)

    dense_feature, sparse_feature, labels = next_element[0], next_element[
        1], next_element[2]
    
    input_features = None

    if args.use_feature_columns:

        feature_columns = transform_feature_column()

        features = transform_features(sparse_feature, dense_feature)

        input_features = tf.feature_column.input_layer(features, feature_columns)
    else:
        
        deep_features = transform_categorical(sparse_feature)

        wide_features = transform_numeric(dense_feature)

        input_features = deep_features + wide_features

    logits = stacked_dcn_v2(features=input_features,
                            mlp_dims=[1024, 512, 256, 1])

    labels = tf.reshape(labels, (-1, 1))

    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits))

    loss = hvd.allreduce(loss, op=hvd.Sum)

    step = tf.train.get_or_create_global_step()

    opt = tf.train.AdagradOptimizer(learning_rate=0.01)

    train_op = opt.minimize(loss, global_step=step)

    # Session config

    sess_config = tf.ConfigProto()

    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())

    sess_config.gpu_options.allow_growth = True

    # # Session hooks

    hooks = []

    # if args.smartstaged and not args.tf:

    #     '''Smart staged Feature'''

    #     next_element = tf.staged(next_element, num_threads=4, capacity=40)

    #     sess_config.graph_options.optimizer_options.do_smart_stage = True

    #     hooks.append(tf.make_prefetch_hook())

    # if args.op_fusion and not args.tf:

    #     '''Auto Graph Fusion'''

    #     sess_config.graph_options.optimizer_options.do_op_fusion = True

    # if args.micro_batch and not args.tf:

    #     '''Auto Mirco Batch'''

    #     sess_config.graph_options.optimizer_options.micro_batch_num = args.micro_batch
    scaffold = tf.train.Scaffold(local_init_op=tf.group(
        tf.local_variables_initializer(), train_init_op))

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    stop_hook = tf.train.StopAtStepHook(last_step=train_steps)

    hooks.append(stop_hook)

    log_hook = tf.train.LoggingTensorHook({
        'steps': step,
        'loss': loss,
    }, every_n_iter=500)
    hooks.append(log_hook)

    with tf.train.MonitoredTrainingSession(master = '',
                                           hooks=hooks,
                                           checkpoint_dir=checkpoint_dir,
                                           scaffold=scaffold,
                                           config=sess_config) as sess:

        while not sess.should_stop():
            sess.run([loss, train_op])
            
    print("Training completed.")


def boolean_string(string):

    low_string = string.lower()

    if low_string not in {'false', 'true'}:

        raise ValueError('Not a valid boolean string')

    return low_string == 'true'


# Get parse


def get_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='./data')

    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=501)

    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)

    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)

    parser.add_argument('--workqueue',
                        help='Whether to enable Work Queue. Default to False.',
                        type=boolean_string,
                        default=False)

    parser.add_argument('--use_feature_columns', action='store_true')    

    return parser


# Some DeepRec's features are enabled by ENV.

# This func is used to set ENV and enable these features.

# A triple quotes comment is used to introduce these features and play an emphasizing role.


def set_env_for_DeepRec():
    '''

    Set some ENV for these DeepRec's features enabled by ENV. 

    More Detail information is shown in https://deeprec.readthedocs.io/zh/latest/index.html.

    START_STATISTIC_STEP & STOP_STATISTIC_STEP: On CPU platform, DeepRec supports memory optimization

        in both stand-alone and distributed trainging. It's default to open, and the 

        default start and stop steps of collection is 1000 and 1100. Reduce the initial 

        cold start time by the following settings.

    MALLOC_CONF: On CPU platform, DeepRec can use memory optimization with the jemalloc library.

        Please preload libjemalloc.so by `LD_PRELOAD=./libjemalloc.so.2 python ...`

    '''

    os.environ['START_STATISTIC_STEP'] = '100'

    os.environ['STOP_STATISTIC_STEP'] = '110'

    os.environ[
        'MALLOC_CONF'] = 'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'


if __name__ == '__main__':

    parser = get_arg_parser()

    args = parser.parse_args()

    set_env_for_DeepRec()

    main()
