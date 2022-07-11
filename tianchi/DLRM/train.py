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

import time
import argparse
import tensorflow as tf
import os
import sys
import math
import collections
import json

from tensorflow.python.feature_column import utils as fc_utils

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
LABEL_COLUMN = ['clicked']
LABEL_COLUMN_DEFAULTS = [0]
USER_COLUMNS = [
    'user_id', 'gender', 'visit_city', 'avg_price', 'is_supervip', 'ctr_30',
    'ord_30', 'total_amt_30'
]
USER_COLUMNS_DEFAULTS = ['', -99, '0', 0.0, 0, 0, 0, 0.0]
ITEM_COLUMN = [
    'shop_id', 'item_id', 'city_id', 'district_id', 'shop_aoi_id',
    'shop_geohash_6', 'shop_geohash_12', 'brand_id', 'category_1_id',
    'merge_standard_food_id', 'rank_7', 'rank_30', 'rank_90'
]
ITEM_COLUMN_DEFAULTS = ['', '', '0', '0', '', '', '', '0', '0', '0', 0, 0, 0]
HISTORY_COLUMN = [
    'shop_id_list', 'item_id_list', 'category_1_id_list',
    'merge_standard_food_id_list', 'brand_id_list', 'price_list',
    'shop_aoi_id_list', 'shop_geohash6_list', 'timediff_list', 'hours_list',
    'time_type_list', 'weekdays_list'
]
HISTORY_COLUMN_DEFAULTS = ['', '', '', '', '', '0', '', '', '0', '-0', '', '']
USER_TZ_COLUMN = ['times', 'hours', 'time_type', 'weekdays', 'geohash12']
USER_TZ_COLUMN_DEFAULTS = ['0', 0, '', 0, '']
DEFAULTS = LABEL_COLUMN_DEFAULTS + USER_COLUMNS_DEFAULTS + ITEM_COLUMN_DEFAULTS + HISTORY_COLUMN_DEFAULTS + USER_TZ_COLUMN_DEFAULTS

FEATURE_COLUMNS = USER_COLUMNS + ITEM_COLUMN + HISTORY_COLUMN + USER_TZ_COLUMN
TRAIN_DATA_COLUMNS = LABEL_COLUMN + FEATURE_COLUMNS
SHARE_EMBEDDING_COLS = [
    ['shop_id', 'shop_id_list'], ['item_id', 'item_id_list'],
    ['category_1_id', 'category_1_id_list'],
    ['merge_standard_food_id', 'merge_standard_food_id_list'],
    ['brand_id', 'brand_id_list'], ['shop_aoi_id', 'shop_aoi_id_list'],
    ['shop_geohash_12', 'geohash12'], ['shop_geohash_6', 'shop_geohash6_list'],
    ['visit_city', 'city_id']
]
EMBEDDING_COLS = ['user_id', 'district_id', 'times', 'timediff_list']
CONTINUOUS_COLUMNS = [
    'gender', 'avg_price', 'is_supervip', 'ctr_30', 'ord_30', 'total_amt_30',
    'rank_7', 'rank_30', 'rank_90', 'hours'
]
CONTINUOUS_HISTORY_COLUMNS = ['price_list', 'hours_list']
TYPE_COLS = ['time_type', 'time_type_list']
TYPE_LIST = ['lunch', 'night', 'dinner', 'tea', 'breakfast']

HASH_BUCKET_SIZES = 100000
EMBEDDING_DIMENSIONS = 16


class DLRM():

    def __init__(self,
                 dense_column=None,
                 sparse_column=None,
                 mlp_bot=[512, 256, 64, 16],
                 mlp_top=[512, 256],
                 optimizer_type='adam',
                 learning_rate=0.1,
                 inputs=None,
                 interaction_op='dot',
                 bf16=False,
                 stock_tf=None,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        self._feature = inputs[0]
        self._label = inputs[1]

        if not dense_column or not sparse_column:
            raise ValueError('Dense column or sparse column is not defined.')
        self._dense_column = dense_column
        self._sparse_column = sparse_column

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True

        self._mlp_bot = mlp_bot
        self._mlp_top = mlp_top
        self._learning_rate = learning_rate
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner
        self._optimizer_type = optimizer_type
        self.interaction_op = interaction_op
        if self.interaction_op not in ['dot', 'cat']:
            print("Invaild interaction op, must be 'dot' or 'cat'.")
            sys.exit()

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

    def _dot_op(self, features):
        batch_size = tf.shape(features)[0]
        matrixdot = tf.matmul(features, features, transpose_b=True)
        feature_dim = matrixdot.shape[-1]

        ones_mat = tf.ones_like(matrixdot)
        lower_tri_mat = ones_mat - tf.linalg.band_part(ones_mat, 0, -1)
        lower_tri_mask = tf.cast(lower_tri_mat, tf.bool)
        result = tf.boolean_mask(matrixdot, lower_tri_mask)
        output_dim = feature_dim * (feature_dim - 1) // 2

        return tf.reshape(result, (batch_size, output_dim))

    # create model
    def _create_model(self):
        # input dense feature and embedding of sparse features
        with tf.variable_scope('input_layer', reuse=tf.AUTO_REUSE):
            for key in HISTORY_COLUMN:
                self._feature[key] = tf.strings.split(self._feature[key], ';')
            for key in CONTINUOUS_HISTORY_COLUMNS:
                length = fc_utils.sequence_length_from_sparse_tensor(
                    self._feature[key])
                length = tf.expand_dims(length, -1)
                self._feature[key] = tf.sparse.to_dense(self._feature[key],
                                                        default_value='0')
                self._feature[key] = tf.strings.to_number(self._feature[key])
                self._feature[key] = tf.reduce_sum(self._feature[key], 1, True)
                self._feature[key] = tf.math.divide(
                    self._feature[key], tf.cast(length, tf.float32))

            with tf.variable_scope('dense_input_layer',
                                   partitioner=self._input_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                dense_inputs = tf.feature_column.input_layer(
                    self._feature, self._dense_column)
            with tf.variable_scope('sparse_input_layer', reuse=tf.AUTO_REUSE):
                column_tensors = {}
                sparse_inputs = tf.feature_column.input_layer(
                    features=self._feature,
                    feature_columns=self._sparse_column,
                    cols_to_output_tensors=column_tensors)

        # MLP behind dense inputs
        mlp_bot_scope = tf.variable_scope(
            'mlp_bot_layer',
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with mlp_bot_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else mlp_bot_scope:
            if self.bf16:
                dense_inputs = tf.cast(dense_inputs, dtype=tf.bfloat16)

            for layer_id, num_hidden_units in enumerate(self._mlp_bot):
                with tf.variable_scope(
                        'mlp_bot_hiddenlayer_%d' % layer_id,
                        reuse=tf.AUTO_REUSE) as mlp_bot_hidden_layer_scope:
                    dense_inputs = tf.layers.dense(
                        dense_inputs,
                        units=num_hidden_units,
                        activation=tf.nn.relu,
                        name=mlp_bot_hidden_layer_scope)
                    dense_inputs = tf.layers.batch_normalization(
                        dense_inputs,
                        training=self.is_training,
                        trainable=True)
                    self._add_layer_summary(dense_inputs,
                                            mlp_bot_hidden_layer_scope.name)
            if self.bf16:
                dense_inputs = tf.cast(dense_inputs, dtype=tf.float32)

        # interaction_op
        if self.interaction_op == 'dot':
            # dot op
            with tf.variable_scope('Op_dot_layer', reuse=tf.AUTO_REUSE):
                mlp_input = [dense_inputs]
                for cols in self._sparse_column:
                    mlp_input.append(column_tensors[cols])
                mlp_input = tf.stack(mlp_input, axis=1)
                mlp_input = self._dot_op(mlp_input)
                mlp_input = tf.concat([dense_inputs, mlp_input], 1)
        elif self.interaction_op == 'cat':
            mlp_input = tf.concat([dense_inputs, sparse_inputs], 1)

        # top MLP before output
        if self.bf16:
            mlp_input = tf.cast(mlp_input, dtype=tf.bfloat16)
        mlp_top_scope = tf.variable_scope(
            'mlp_top_layer',
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with mlp_top_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else mlp_top_scope:
            for layer_id, num_hidden_units in enumerate(self._mlp_top):
                with tf.variable_scope(
                        'mlp_top_hiddenlayer_%d' % layer_id,
                        reuse=tf.AUTO_REUSE) as mlp_top_hidden_layer_scope:
                    mlp_logits = tf.layers.dense(
                        mlp_input,
                        units=num_hidden_units,
                        activation=tf.nn.relu,
                        name=mlp_top_hidden_layer_scope)

                self._add_layer_summary(mlp_logits,
                                        mlp_top_hidden_layer_scope.name)

        if self.bf16:
            mlp_logits = tf.cast(mlp_logits, dtype=tf.float32)

        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE) as logits_scope:
            self._logits = tf.layers.dense(mlp_logits,
                                           units=1,
                                           activation=None,
                                           name=logits_scope)
            self.probability = tf.math.sigmoid(self._logits)
            self.output = tf.round(self.probability)

            self._add_layer_summary(self.probability, logits_scope.name)

    # compute loss

    def _create_loss(self):
        loss_func = tf.keras.losses.BinaryCrossentropy()
        predict = tf.squeeze(self.probability)
        self.loss = tf.math.reduce_mean(loss_func(self._label, predict))
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
        elif self._optimizer_type == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._learning_rate,
                initial_accumulator_value=0.1,
                use_locking=False)
        elif self._optimizer_type == 'gradientdescent':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self._learning_rate)
        else:
            raise ValueError("Optimzier type error.")

        self.train_op = optimizer.minimize(self.loss,
                                           global_step=self.global_step)

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
        column_headers = TRAIN_DATA_COLUMNS
        columns = tf.io.decode_csv(value, record_defaults=DEFAULTS)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

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


# generate feature columns
def build_feature_columns():
    dense_column = []
    sparse_column = []
    for columns in SHARE_EMBEDDING_COLS:
        cate_cols = []
        for col in columns:
            cate_col = tf.feature_column.categorical_column_with_hash_bucket(
                col, HASH_BUCKET_SIZES)
            cate_cols.append(cate_col)
        shared_emb_cols = tf.feature_column.shared_embedding_columns(
            cate_cols, EMBEDDING_DIMENSIONS)
        sparse_column.extend(shared_emb_cols)

    for column in EMBEDDING_COLS:
        cate_col = tf.feature_column.categorical_column_with_hash_bucket(
            column, HASH_BUCKET_SIZES)

        if args.tf or not args.emb_fusion:
            emb_col = tf.feature_column.embedding_column(
                cate_col, EMBEDDING_DIMENSIONS)
        else:
            emb_col = tf.feature_column.embedding_column(
                cate_col, EMBEDDING_DIMENSIONS, do_fusion=args.emb_fusion)
        sparse_column.append(emb_col)

    for column in CONTINUOUS_COLUMNS + CONTINUOUS_HISTORY_COLUMNS:
        num_column = tf.feature_column.numeric_column(column)
        dense_column.append(num_column)

    for column in TYPE_COLS:
        cate_col = tf.feature_column.categorical_column_with_vocabulary_list(
            column, TYPE_LIST)
        if args.tf or not args.emb_fusion:
            emb_col = tf.feature_column.embedding_column(
                cate_col, EMBEDDING_DIMENSIONS)
        else:
            emb_col = tf.feature_column.embedding_column(
                cate_col, EMBEDDING_DIMENSIONS, do_fusion=args.emb_fusion)
        sparse_column.append(emb_col)

    return dense_column, sparse_column


def train(sess_config,
          input_hooks,
          model,
          data_init_op,
          steps,
          checkpoint_dir,
          tf_config=None,
          server=None):
    model.is_training = True
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.tables_initializer(),
                               tf.local_variables_initializer(), data_init_op),
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

    with tf.train.MonitoredTrainingSession(
            master=server.target if server else '',
            is_chief=tf_config['is_chief'] if tf_config else True,
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
        local_init_op=tf.group(tf.tables_initializer(),
                               tf.local_variables_initializer(), data_init_op))
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
    writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval'))
    merged = tf.summary.merge_all()

    with tf.train.MonitoredSession(session_creator=session_creator,
                                   hooks=hooks) as sess:
        for _in in range(1, steps + 1):
            if (_in != steps):
                sess.run([model.acc_op, model.auc_op])
                if (_in % 100 == 0):
                    print("Evaluation complate:[{}/{}]".format(_in, steps))
            else:
                eval_acc, eval_auc, events = sess.run(
                    [model.acc_op, model.auc_op, merged])
                writer.add_summary(events, _in)
                print("Evaluation complate:[{}/{}]".format(_in, steps))
                print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))


def main(tf_config=None, server=None):
    # check dataset and count data set size
    print("Checking dataset...")
    train_file = os.path.join(args.data_location, 'train.csv')
    test_file = os.path.join(args.data_location, 'eval.csv')
    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print("Dataset does not exist in the given data_location.")
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))

    # set batch size, eporch & steps
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
                             'model_DLRM_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
    test_dataset = build_model_input(test_file, batch_size, 1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create feature column
    dense_column, sparse_column = build_feature_columns()

    # Session config
    sess_config = tf.ConfigProto()

    # Session hooks
    hooks = []

    if args.smartstaged and not args.tf:
        '''Smart staged Feature'''
        next_element = tf.staged(next_element, num_threads=4, capacity=40)
        sess_config.graph_options.optimizer_options.do_smart_stage = True
        hooks.append(tf.make_prefetch_hook())
    if args.op_fusion and not args.tf:
        '''Auto Graph Fusion'''
        sess_config.graph_options.optimizer_options.do_op_fusion = True

    # create model
    model = DLRM(dense_column=dense_column,
                 sparse_column=sparse_column,
                 learning_rate=args.learning_rate,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=args.tf,
                 interaction_op=args.interaction_op,
                 inputs=next_element)

    # Run model training and evaluation
    train(sess_config, hooks, model, train_init_op, train_steps,
          checkpoint_dir, tf_config, server)
    if not (args.no_eval or tf_config):
        eval(sess_config, hooks, model, test_init_op, test_steps,
             checkpoint_dir)


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
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output directory',
                        required=False)
    parser.add_argument('--deep_dropout',
                        help='Dropout regularization for deep model',
                        type=float,
                        default=0.0)
    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)
    parser.add_argument('--optimizer', \
                        type=str,
                        choices=['adam', 'adamasync', 'adagraddecay',
                                 'adagrad', 'gradientdescent'],
                        default='adamasync')
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.01)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--interaction_op',
                        type=str,
                        choices=['dot', 'cat'],
                        default='cat')
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc', 'grpc++', 'star_server'],
                        default='grpc')
    parser.add_argument('--tf',
                        help='Use TF 1.15.5 API and disable DeepRec feature.',
                        action='store_true')
    parser.add_argument('--smartstaged', \
                        help='Whether to enable smart staged feature of DeepRec.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--emb_fusion', \
                        help='Whether to enable embedding fusion, Default to True.',
                        type=boolean_string,
                        default=True)
    parser.add_argument('--op_fusion',
                        help='Whether to enable Auto graph fusion feature.',
                        type=boolean_string,
                        default=True)
    return parser


# Parse distributed training configuration and generate cluster information
def generate_cluster_info(TF_CONFIG):
    print(TF_CONFIG)
    tf_config = json.loads(TF_CONFIG)
    cluster_config = tf_config.get('cluster')
    ps_hosts = []
    worker_hosts = []
    chief_hosts = []
    for key, value in cluster_config.items():
        if 'ps' == key:
            ps_hosts = value
        elif 'worker' == key:
            worker_hosts = value
        elif 'chief' == key:
            chief_hosts = value
    if chief_hosts:
        worker_hosts = chief_hosts + worker_hosts

    if not ps_hosts or not worker_hosts:
        print('TF_CONFIG ERROR')
        sys.exit()
    task_config = tf_config.get('task')
    task_type = task_config.get('type')
    task_index = task_config.get('index') + (1 if task_type == 'worker'
                                             and chief_hosts else 0)

    if task_type == 'chief':
        task_type = 'worker'

    is_chief = True if task_index == 0 else False
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.distribute.Server(cluster,
                                  job_name=task_type,
                                  task_index=task_index,
                                  protocol=args.protocol)
    if task_type == 'ps':
        server.join()
    elif task_type == 'worker':
        tf_config = {
            'ps_hosts': ps_hosts,
            'worker_hosts': worker_hosts,
            'type': task_type,
            'index': task_index,
            'is_chief': is_chief
        }
        tf_device = tf.device(
            tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % task_index,
                cluster=cluster))
        return tf_config, server, tf_device
    else:
        print("Task type or index error.")
        sys.exit()


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

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        main()
    else:
        tf_config, server, tf_device = generate_cluster_info(TF_CONFIG)
        main(tf_config, server)
