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

import argparse
import collections
import horovod.tensorflow as hvd
import math
import numpy as np
import os
import sys
import time
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.feature_column import feature_column as fc


# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ['clicked']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

def transform_categorical(feature):
    max_value = np.iinfo(tf.uint32.as_numpy_dtype).max
    variables = []
    indices = []
    deep_features = []
    for i, column_name in enumerate(CATEGORICAL_COLUMNS):
        if args.tf or not args.ev:
            embedding_weights = tf.get_variable(
                f'{column_name}_weight',
                shape=[max_value, 16],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05))
        else:
            '''Feature Elimination of EmbeddingVariable Feature'''
            if args.ev_elimination == 'gstep':
                # Feature elimination based on global steps
                evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
            elif args.ev_elimination == 'l2':
                # Feature elimination based on l2 weight
                evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)
            else:
                evict_opt = None
                '''Feature Filter of EmbeddingVariable Feature'''
                if args.ev_filter == 'cbf':
                    # CBF-based feature filter
                    filter_option = tf.CBFFilter(
                        filter_freq=3,
                        max_element_size=2**30,
                        false_positive_probability=0.01,
                        counter_type=tf.int64)
                elif args.ev_filter == 'counter':
                    # Counter-based feature filter
                    filter_option = tf.CounterFilter(filter_freq=3)
                else:
                    filter_option = None
            ev_opt = tf.EmbeddingVariableOption(
                evict_option=evict_opt, filter_option=filter_option)            

            embedding_weights = tf.get_embedding_variable(
                f'{column_name}_weight',
                initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.05),
                embedding_dim=16,
                ev_option=ev_opt)
            
        i = CATEGORICAL_COLUMNS.index(column_name)
        if args.enable_hvd:
            target_gpu = -1
            embedding_weights.target_gpu = target_gpu

        if not args.tf and args.group_embedding:
            if args.enable_hvd:
                feature_tensor = tf.RaggedTensor.from_row_lengths(
                    values=feature[i], row_lengths=tf.ones_like(feature[i]))
            else:
                feature_tensor = fc._to_sparse_input_and_drop_ignore_values(feature[i])
                feature_tensor = tf.sparse.reshape(feature_tensor, (-1, 1))
            indices.append(feature_tensor)
            variables.append(embedding_weights)
        else:
            sparse_tensor = fc._to_sparse_input_and_drop_ignore_values(feature[i])
            sparse_tensor = tf.sparse.reshape(sparse_tensor, (-1, 1))
            deep_feature = tf.nn.safe_embedding_lookup_sparse(
                embedding_weights=embedding_weights,
                sparse_ids=sparse_tensor,
                combiner='sum')
            deep_features.append(deep_feature)
            

    if args.group_embedding:
        combiners = ['sum' for _ in range(len(CATEGORICAL_COLUMNS))]
        deep_features = tf.nn.group_embedding_lookup_sparse(variables, indices, combiners)

    return deep_features

def transform_numeric(feature):
    r'''Transform numeric features.

    '''
    numeric_list = []
    for column_name in CONTINUOUS_COLUMNS:
        i = CONTINUOUS_COLUMNS.index(column_name)
        numeric_list.append(tf.reshape(feature[i], shape=[-1, 1]))

    return numeric_list

class DeepFM():
    def __init__(self,
                 inputs=None,
                 dnn_hidden_units=[1024, 256, 32],
                 final_hidden_units=[128, 64],
                 optimizer_type='adam',
                 learning_rate=0.001,
                 use_bn=True,
                 bf16=False,
                 stock_tf=None,
                 enable_hvd=False):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        self._dense_features = inputs[0]
        self._sparse_features = inputs[1]
        self._label = inputs[2]

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True
        self.use_bn = use_bn
        self.enable_hvd = enable_hvd

        self._dnn_hidden_units = dnn_hidden_units
        self._final_hidden_units = final_hidden_units
        self._optimizer_type = optimizer_type
        self._learning_rate = learning_rate

        self._create_model()
        with tf.name_scope('head'):
            self._create_loss()
            self._create_optimizer()
            self._create_metrics()

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    def _dnn(self, dnn_input, dnn_hidden_units=None, layer_name=''):
        for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with tf.variable_scope(layer_name + '_%d' % layer_id) as dnn_layer_scope:
                dnn_input = tf.layers.dense(
                    dnn_input,
                    units=num_hidden_units,
                    activation=tf.nn.relu,
                    name=dnn_layer_scope)
                if self.use_bn:
                    dnn_input = tf.layers.batch_normalization(
                        dnn_input, training=self.is_training, trainable=True)
                self._add_layer_summary(dnn_input, dnn_layer_scope.name)

        return dnn_input

    def _create_model(self):
        # input features
        with tf.variable_scope('input_layer'):
            self._dense_input = transform_numeric(self._dense_features)
            dense_input = tf.concat(self._dense_input, axis=1)
            self._sparse_input = transform_categorical(self._sparse_features)
            sparse_input = tf.concat(self._sparse_input, axis=1)
            dnn_input = tf.concat([dense_input, sparse_input], axis=1)
            wide_input = tf.concat([dense_input, sparse_input], axis=1)
            fm_input = tf.stack(self._sparse_input, axis=1)

        if self.bf16:
            wide_input = tf.cast(wide_input, dtype=tf.bfloat16)
            fm_input = tf.cast(fm_input, dtype=tf.bfloat16)
            dnn_input = tf.cast(dnn_input, dtype=tf.bfloat16)

        # DNN part
        dnn_scope = tf.variable_scope('dnn')
        with dnn_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else dnn_scope:
            dnn_output = self._dnn(dnn_input, self._dnn_hidden_units,
                                   'dnn_layer')

        # linear / fisrt order part
        with tf.variable_scope('linear', reuse=tf.AUTO_REUSE) as linear:
            linear_output = tf.reduce_sum(wide_input, axis=1, keepdims=True)

        # FM second order part
        with tf.variable_scope('fm', reuse=tf.AUTO_REUSE) as fm:
            sum_square = tf.square(tf.reduce_sum(fm_input, axis=1))
            square_sum = tf.reduce_sum(tf.square(fm_input), axis=1)
            fm_output = 0.5 * tf.subtract(sum_square, square_sum)

        # Final dnn layer
        all_input = tf.concat([dnn_output, linear_output, fm_output], 1)
        final_dnn_scope = tf.variable_scope('final_dnn')
        with final_dnn_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else final_dnn_scope:
            dnn_logits = self._dnn(all_input, self._final_hidden_units, 'final_dnn')

        if self.bf16:
            dnn_logits = tf.cast(dnn_logits, dtype=tf.float32)

        self._logits = tf.layers.dense(dnn_logits, units=1)
        self.probability = tf.math.sigmoid(self._logits)
        self.output = tf.round(self.probability)

    # compute loss
    def _create_loss(self):
        loss_func = tf.losses.mean_squared_error
        predict = tf.squeeze(self.probability)
        self.loss = tf.math.reduce_mean(loss_func(self._label, predict))
        if self.enable_hvd:
            self.loss = hvd.allreduce(self.loss, op=hvd.Sum)
        tf.summary.scalar('loss', self.loss)

    # define optimizer and generate train_op
    def _create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        if self.tf or self._optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._learning_rate,
                initial_accumulator_value=1e-8)
        elif self._optimizer_type == 'adamasync':
            optimizer = tf.train.AdamAsyncOptimizer(
                learning_rate=self._learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagraddecay':
            optimizer = tf.train.AdagradDecayOptimizer(
                learning_rate=self._learning_rate,
                global_step=self.global_step)
        else:
            raise ValueError('Optimizer type error.')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(
                self.loss, global_step=self.global_step)

    # compute acc & auc
    def _create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self._label,
                                                    predictions=self.output)
        self.auc, self.auc_op = tf.metrics.auc(labels=self._label,
                                               predictions=self.probability,
                                               num_thresholds=1000)
        tf.summary.scalar('eval_acc', self.acc)
        tf.summary.scalar('eval_auc', self.auc)


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        cont_defaults = [[0.0] for i in range(1, 14)]
        cate_defaults = [[tf.constant(0, dtype=tf.int64)] for i in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cont_defaults + cate_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        dense_features = [all_columns[name] for name in CONTINUOUS_COLUMNS]
        sparse_features = [all_columns[name] for name in CATEGORICAL_COLUMNS]
        labels = [all_columns[name] for name in LABEL_COLUMN]
        return dense_features, sparse_features, labels
    
    def parse_parquet(value):
        tf.logging.info('Parsing {}'.format(filename))
        dense_features = [value[name] for name in CONTINUOUS_COLUMNS]
        sparse_features = [value[name] for name in CATEGORICAL_COLUMNS]
        labels = [value[name] for name in LABEL_COLUMN]
        return dense_features, sparse_features, labels

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
    if args.parquet_dataset:
        from tensorflow.python.data.experimental.ops import parquet_dataset_ops
        dataset = parquet_dataset_ops.ParquetDataset(files, batch_size=batch_size)
        if args.parquet_dataset_shuffle:
            dataset = dataset.shuffle(buffer_size=20000,
                                      seed=args.seed)  # fix seed for reproducing
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(parse_parquet, num_parallel_calls=28)
    else:
        dataset = tf.data.TextLineDataset(files)
        dataset = dataset.shuffle(buffer_size=20000,
                                  seed=args.seed)  # fix seed for reproducing
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(2)
    return dataset


def train(sess_config,
          input_hooks,
          model,
          data_init_op,
          steps,
          checkpoint_dir):
    model.is_training = True
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), data_init_op),
        saver=tf.train.Saver(max_to_keep=args.keep_checkpoint_max))

    stop_hook = tf.train.StopAtStepHook(last_step=steps)
    log_hook = tf.train.LoggingTensorHook(
        {
            'steps': model.global_step,
            'loss': model.loss
        }, every_n_iter=100)
    hooks.append(stop_hook)
    hooks.append(log_hook)
    if args.timeline > 0:
        hooks.append(
            tf.train.ProfilerHook(save_steps=args.timeline,
                                  output_dir=checkpoint_dir))
    save_steps = args.save_steps if args.save_steps or args.no_eval else steps
    '''
                            Incremental_Checkpoint
    Please add `save_incremental_checkpoint_secs` in 'tf.train.MonitoredTrainingSession'
    it's default to None, Incremental_save checkpoint time in seconds can be set 
    to use incremental checkpoint function, like `tf.train.MonitoredTrainingSession(
        save_incremental_checkpoint_secs=args.incremental_ckpt)`
    '''
    if args.incremental_ckpt and not args.tf:
        print("Incremental_Checkpoint is not really enabled.")
        print("Please see the comments in the code.")
        sys.exit()

    with tf.train.MonitoredTrainingSession(
            hooks=hooks,
            scaffold=scaffold,
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=save_steps,
            summary_dir=checkpoint_dir,
            save_summaries_steps=args.save_steps,
            config=sess_config) as sess:
        while not sess.should_stop():
            sess.run([model.loss, model.train_op])
    print("Training completed.")


def eval(sess_config, input_hooks, model, data_init_op, steps, checkpoint_dir):
    model.is_training = False
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), data_init_op))
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
    writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval'))
    merged = tf.summary.merge_all()

    with tf.train.MonitoredSession(session_creator=session_creator,
                                   hooks=hooks) as sess:
        for _in in range(1, steps + 1):
            if (_in != steps):
                sess.run([model.acc_op, model.auc_op])
                if (_in % 1000 == 0):
                    print("Evaluation complete:[{}/{}]".format(_in, steps))
            else:
                eval_acc, eval_auc, events = sess.run(
                    [model.acc_op, model.auc_op, merged])
                writer.add_summary(events, _in)
                print("Evaluation complete:[{}/{}]".format(_in, steps))
                print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))


def main():
    # check dataset
    print('Checking dataset')
    train_file = args.data_location
    test_file = args.data_location
    if args.parquet_dataset:
        train_file += '/train.parquet'
        test_file += '/eval.parquet'
    else:
        train_file += '/train.csv'
        test_file += '/eval.csv'
    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print("Dataset does not exist in the given data_location.")
        sys.exit()
    no_of_training_examples = 0
    no_of_test_examples = 0
    if args.parquet_dataset:
        import pyarrow.parquet as pq
        no_of_training_examples = pq.read_table(train_file).num_rows
        no_of_test_examples = pq.read_table(test_file).num_rows
    else:
        no_of_training_examples = sum(1 for line in open(train_file))
        no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of training dataset is {}".format(no_of_training_examples))
    print("Numbers of test dataset is {}".format(no_of_test_examples))

    # set batch size, eporch & steps
    batch_size = 0
    if args.micro_batch and not args.tf:
        batch_size = math.ceil(args.batch_size / args.micro_batch)
    elif args.enable_hvd:
        assert args.batch_size % hvd.size() == 0
        batch_size = int(args.batch_size / hvd.size())
    else:
        batch_size = args.batch_size

    if args.steps == 0:
        no_of_epochs = 1
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)
    print("The training steps is {}".format(train_steps))
    print("The testing steps is {}".format(test_steps))

    # set fixed random seed
    tf.set_random_seed(args.seed)

    # set directory path
    model_dir = os.path.join(args.output_dir,
                             'model_DeepFM_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
    test_dataset = build_model_input(test_file, batch_size, 1)

    dataset_output_types = tf.data.get_output_types(train_dataset)
    dataset_output_shapes = tf.data.get_output_shapes(test_dataset)
    iterator = tf.data.Iterator.from_structure(dataset_output_types,
                                               dataset_output_shapes)    
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)


    # Session config
    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    # Session hooks
    hooks = []
    if args.enable_hvd:
        sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
    if args.smartstaged and not args.tf:
        '''Smart staged Feature'''
        next_element = tf.staged(next_element, num_threads=2, capacity=16)
        sess_config.graph_options.optimizer_options.do_smart_stage = True
        sess_config.graph_options.optimizer_options.stage_subgraph_on_cpu = True
        hooks.append(tf.make_prefetch_hook())
    if args.op_fusion and not args.tf:
        '''Auto Graph Fusion'''
        sess_config.graph_options.optimizer_options.do_op_fusion = True
    if args.micro_batch and not args.tf:
        '''Auto Mirco Batch'''
        sess_config.graph_options.optimizer_options.micro_batch_num = args.micro_batch

    # create model
    model = DeepFM(inputs=next_element,
                   optimizer_type=args.optimizer,
                   learning_rate=args.learning_rate,
                   bf16=args.bf16,
                   stock_tf=args.tf,
                   enable_hvd=args.enable_hvd)

    # Run model training and evaluation
    train(sess_config, hooks, model, train_init_op, train_steps, checkpoint_dir)
    if not (args.no_eval):
        eval(sess_config, hooks, model, test_init_op, test_steps, checkpoint_dir)

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
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 16384',
                        type=int,
                        default=8192)
    parser.add_argument('--output_dir',
                        help='Full path to model output directory. \
                            Default to ./result. Covered by --checkpoint. ',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output. \
                            Default to ./result/$MODEL_TIMESTAMP',
                        required=False)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)
    parser.add_argument('--optimizer',
                        type=str,
                        choices=['adam', 'adamasync',
                                 'adagraddecay', 'adagrad'],
                        default='adamasync')
    parser.add_argument('--learning_rate',
                        help='Learning rate for deep model',
                        type=float,
                        default=0.001)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--tf',
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--smartstaged',
                        help='Whether to enable smart staged feature of DeepRec, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--ev',
                        help='Whether to enable DeepRec EmbeddingVariable. Default True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--ev_elimination',
                        help='Feature Elimination of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter',
                        help='Feature Filter of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--op_fusion',
                        help='Whether to enable Auto graph fusion feature. Default to True',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--micro_batch',
                        help='Set num for Auto Mirco Batch. Default close.',
                        type=int,
                        default=0) # TODO enable # TODO enable
    parser.add_argument('--incremental_ckpt',
                        help='Set time of save Incremental Checkpoint. Default 0 to close.',
                        type=boolean_string,
                        default=0)
    parser.add_argument('--workqueue',
                        help='Whether to enable Work Queue. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument("--parquet_dataset", \
                        help='Whether to enable Parquet DataSet. Defualt to False.',
                        type=boolean_string,
                        default=True)
    parser.add_argument("--parquet_dataset_shuffle", \
                        help='Whether to enable shuffle operation for Parquet Dataset. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument("--group_embedding", \
                        help='Whether to enable Group Embedding. Defualt to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument("--enable_hvd", \
                        help="Whether to enable Horovod. Default to False.",
                        type=boolean_string,
                        default=False)

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
    os.environ['MALLOC_CONF'] = \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    if not args.tf:
        set_env_for_DeepRec()

    if args.group_embedding and args.enable_hvd:
            tf.config.experimental.enable_distributed_strategy(strategy="collective")

    main()
