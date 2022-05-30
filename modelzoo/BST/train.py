import time
import argparse
import tensorflow as tf
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
import json

from tensorflow.python.ops import partitioned_variables

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))
INPUT_COLUMN = [
    'clk', 'buy', 'pid', 'adgroup_id', 'cate_id', 'campaign_id', 'customer',
    'brand', 'user_id', 'cms_segid', 'cms_group_id', 'final_gender_code',
    'age_level', 'pvalue_level', 'shopping_level', 'occupation',
    'new_user_class_level', 'tag_category_list', 'tag_brand_list', 'price'
]
USER_COLUMN = [
    'user_id', 'cms_segid', 'cms_group_id', 'age_level', 'pvalue_level',
    'shopping_level', 'occupation', 'new_user_class_level'
]
ITEM_COLUMN = [
    'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'price', 'pid'
]
LABEL_COLUMN = ['clk']
BUY_COLUMN = ['buy']
TAG_COLUMN = ['tag_category_list', 'tag_brand_list']
KEY_COLUMN = ['cate_id', 'brand']
INPUT_FEATURES = {
    'pid': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 10
    },
    'adgroup_id': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 100000
    },
    'cate_id': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 10000
    },
    'campaign_id': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 100000
    },
    'customer': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 100000
    },
    'brand': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 100000
    },
    'user_id': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 100000
    },
    'cms_segid': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 100
    },
    'cms_group_id': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 100
    },
    'final_gender_code': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 10
    },
    'age_level': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 10
    },
    'pvalue_level': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 10
    },
    'shopping_level': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 10
    },
    'occupation': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 10
    },
    'new_user_class_level': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 10
    },
    'tag_category_list': {
        'type': 'SequenceFeature',  # TagFeature 
        'dim': 16,
        'hash_bucket_size': 100000
    },
    'tag_brand_list': {
        'type': 'SequenceFeature',
        'dim': 16,
        'hash_bucket_size': 100000
    },
    'price': {
        'type': 'IdFeature',
        'dim': 16,
        'hash_bucket_size': 50
    },
}


def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def generate_input_data(filename, batch_size, num_epochs):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        string_defaults = [[" "] for i in range(1, 19)]
        label_defaults = [[0], [0]]
        column_headers = INPUT_COLUMN
        record_defaults = label_defaults + string_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        all_columns.pop(BUY_COLUMN[0])
        features = all_columns
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.shuffle(buffer_size=10000,
                              seed=2021)  # set seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(32)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(1)
    return dataset


# a list of features processed via tf.feature_column
def build_feature_cols():
    user_column = []
    item_column = []
    tag_column = []
    key_column = []
    for key in INPUT_FEATURES:
        if key in TAG_COLUMN:
            # parse_sequence_feature
            categorical_column = tf.feature_column.sequence_categorical_column_with_hash_bucket(
                key,
                hash_bucket_size=INPUT_FEATURES[key]['hash_bucket_size'],
                dtype=tf.string)
        else:
            # parse_id_feature
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                key,
                hash_bucket_size=INPUT_FEATURES[key]['hash_bucket_size'],
                dtype=tf.string)
        embedding_column = tf.feature_column.embedding_column(
            categorical_column,
            dimension=INPUT_FEATURES[key]['dim'],
            combiner='mean')
        if key in USER_COLUMN:
            user_column.append(embedding_column)
        elif key in ITEM_COLUMN:
            item_column.append(embedding_column)
            if key in KEY_COLUMN:
                key_column.append(embedding_column)
        elif key in TAG_COLUMN:
            tag_column.append(embedding_column)

    return user_column, item_column, tag_column, key_column


# more to do: dropout//bf16/embedding_regularizations/leaky-relu
class BST():
    def __init__(self,
                 user_column=None,
                 item_column=None,
                 tag_column=None,
                 key_column=None,
                 final_hidden_units=[128, 96, 64, 32, 16],
                 learning_rate=0.001,
                 seqence_length=50,
                 multi_head_size=4,
                 batch_size=0,
                 use_bn=True,
                 l2_regularization=None,
                 beta1=0.9,
                 beta2=0.999,
                 inputs=None,
                 bf16=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        self.unseq_column = user_column + item_column
        self.tag_column = tag_column
        self.key_column = key_column
        self.batch_size = batch_size
        if self.batch_size == 0:
            raise ValueError('Batch size must not be 0.')
        if not user_column or not item_column or not tag_column:
            raise ValueError(
                'User column or item column or tag column is not defined.')
        self.use_bn = use_bn
        self.learning_rate = learning_rate
        self.input_layer_partitioner = input_layer_partitioner
        self.dense_layer_partitioner = dense_layer_partitioner

        self.l2_regularization = l2_regularization
        self.feature = inputs[0]
        self.label = inputs[1]
        self.beta1 = beta1
        self.beta2 = beta2
        self.bf16 = bf16

        self.final_dnn_hidden_units = final_hidden_units
        self._num_class = 1

        self.seq_len = seqence_length
        self.multi_head_size = multi_head_size

        self._is_training = True

        self.predict = self.prediction()
        with tf.name_scope('head'):
            self.train_op, self.loss = self.optimizer()
            self.acc, self.acc_op = tf.metrics.accuracy(labels=self.label,
                                                        predictions=tf.round(
                                                            self.predict))
            self.auc, self.auc_op = tf.metrics.auc(labels=self.label,
                                                   predictions=self.predict,
                                                   num_thresholds=1000)
            tf.summary.scalar('eval_acc', self.acc)
            tf.summary.scalar('eval_auc', self.auc)

    def evalmodel(self):
        self._is_training = False

    def attention_net(self, net, dim, cur_seq_len, seq_size, name):
        # Q K V
        query_net = tf.layers.dense(net,
                                    units=dim,
                                    activation=tf.nn.relu,
                                    name=name + '_query')  # B, seq_len，dim
        key_net = tf.layers.dense(net,
                                  units=dim,
                                  activation=tf.nn.relu,
                                  name=name + '_key')
        value_net = tf.layers.dense(net,
                                    units=dim,
                                    activation=tf.nn.relu,
                                    name=name + '_value')
        # scores = Q K^T
        scores = tf.matmul(query_net, key_net,
                           transpose_b=True)  # [B, seq_size, seq_size]

        # mask: current_max_sequence, sequence_size = 50 (recover from padding)

        # cur_seq_len is tensor scalar, maxlen is int
        # only mask the cur_seq_len part
        hist_mask = tf.sequence_mask(cur_seq_len,
                                     maxlen=seq_size - 1)  # [B, seq_size-1]

        cur_id_mask = tf.ones([tf.shape(hist_mask)[0], 1],
                              dtype=tf.bool)  # [B, 1]
        mask = tf.concat([hist_mask, cur_id_mask], axis=1)  # [B, seq_size]
        masks = tf.reshape(tf.tile(mask, [1, seq_size]),
                           (-1, seq_size, seq_size))  # [B, seq_size, seq_size]
        padding = tf.ones_like(scores) * (-2**32 + 1)
        scores = tf.where(masks, scores, padding)  # [B, seq_size, seq_size]

        # Scale
        if self.bf16:
            scores = tf.cast(scores, dtype=tf.float32)
        scores = tf.nn.softmax(scores)  # (B, seq_size, seq_size)
        if self.bf16:
            scores = tf.cast(scores, dtype=tf.bfloat16)
        att_res_net = tf.matmul(scores, value_net)  # [B, seq_size, emb_dim]
        return att_res_net

    def multi_head_att_net(self,
                           id_cols,
                           head_count,
                           emb_dim,
                           seq_len,
                           seq_size,
                           name='multi_head_att_net'):
        with tf.variable_scope(name):
            multi_head_attention_res = []
            # spilt input to different head
            # emb_dim is current_batch_max_seq_size
            part_cols_emd_dim = int(math.ceil(emb_dim / head_count))
            for start_idx in range(0, emb_dim, part_cols_emd_dim):
                # handle the last head
                if start_idx + part_cols_emd_dim > emb_dim:
                    part_cols_emd_dim = emb_dim - start_idx
                # split input(id_cols)
                part_id_col = tf.slice(id_cols, [0, 0, start_idx],
                                       [-1, -1, part_cols_emd_dim])
                # attention part
                part_attention_net = self.attention_net(part_id_col,
                                                        part_cols_emd_dim,
                                                        seq_len,
                                                        seq_size,
                                                        name='multi_head_%d' %
                                                        start_idx)
                # concate result
                multi_head_attention_res.append(part_attention_net)
            multi_head_attention_res_net = tf.concat(multi_head_attention_res,
                                                     axis=2)
            multi_head_attention_res_net = tf.layers.dense(
                multi_head_attention_res_net,
                units=emb_dim,
                activation=tf.nn.relu,
                name='multi_head_attention')

        return multi_head_attention_res_net

    def add_and_norm(self, net_1, net_2, emb_dim, name='add_and_norm'):
        with tf.variable_scope(name):
            net = tf.add(net_1, net_2)
            layer = tf.keras.layers.LayerNormalization(
                axis=2)  # normalize in embedding dimension
            # layer = layer_norm.LayerNormalization(emb_dim)
            net = layer(net)
        return net

    def multihead_attention(self,
                            id_cols,
                            head_count,
                            emb_dim,
                            seq_len,
                            seq_size,
                            name='multi_head_att_net'):
        with tf.variable_scope(name):
            # emb_dim is current_batch_max_seq_size
            # Q K V
            query_net = tf.layers.dense(id_cols,
                                        units=emb_dim,
                                        activation=tf.nn.relu,
                                        name=name + '_query')  # B, seq_len，dim
            key_net = tf.layers.dense(id_cols,
                                      units=emb_dim,
                                      activation=tf.nn.relu,
                                      name=name + '_key')
            value_net = tf.layers.dense(id_cols,
                                        units=emb_dim,
                                        activation=tf.nn.relu,
                                        name=name + '_value')

            query_net = tf.concat(tf.split(query_net, head_count, axis=-1),
                                  axis=0)
            key_net = tf.concat(tf.split(key_net, head_count, axis=-1), axis=0)
            value_net = tf.concat(tf.split(value_net, head_count, axis=-1),
                                  axis=0)

            # scores = Q K^T
            scores = tf.matmul(query_net, key_net,
                               transpose_b=True)  # [B, seq_size, seq_size]

            # mask: current_max_sequence, sequence_size = 50 (recover from padding)
            # only mask the cur_seq_len part
            hist_mask = tf.sequence_mask(seq_len, maxlen=seq_size -
                                         1)  # [B, seq_size-1]

            cur_id_mask = tf.ones([tf.shape(hist_mask)[0], 1],
                                  dtype=tf.bool)  # [B, 1]
            mask = tf.concat([hist_mask, cur_id_mask], axis=1)  # [B, seq_size]
            masks = tf.reshape(
                tf.tile(mask, [head_count, seq_size]),
                (-1, seq_size, seq_size))  # [B, seq_size, seq_size]

            padding = tf.ones_like(scores) * (-2**32 + 1)
            scores = tf.where(masks, scores,
                              padding)  # [B, seq_size, seq_size]

            # Scale
            if self.bf16:
                scores = tf.cast(scores, dtype=tf.float32)
            scores = tf.nn.softmax(scores)  # (B, seq_size, seq_size)
            if self.bf16:
                scores = tf.cast(scores, dtype=tf.bfloat16)
            att_res_net = tf.matmul(scores,
                                    value_net)  # [B, seq_size, emb_dim]

            att_res_net = tf.concat(tf.split(att_res_net, head_count, axis=0),
                                    axis=2)

            att_res_net = tf.layers.dense(att_res_net,
                                          units=emb_dim,
                                          activation=tf.nn.relu,
                                          name='multi_head_attention')

        return att_res_net

    def bst(self, bst_fea, seq_size, head_count, name):
        cur_id, hist_id_col, seq_len = bst_fea['key'], bst_fea[
            'hist_seq_emb'], bst_fea['hist_seq_len']
        cur_batch_max_seq_len = tf.shape(hist_id_col)[1]
        # sequence padding/slice in the dimension of sequence_size
        # hist_id_col: (batch_size, max_sequence_size, embedding_size)
        hist_id_col = tf.cond(
            tf.constant(seq_size) > cur_batch_max_seq_len,
            lambda: tf.pad(hist_id_col,
                           [[0, 0], [0, seq_size - cur_batch_max_seq_len - 1],
                            [0, 0]], 'CONSTANT'),
            lambda: tf.slice(hist_id_col, [0, 0, 0], [-1, seq_size - 1, -1]))
        # expand to same shape
        # (batch_size, seq_size, emb_dim)
        all_ids = tf.concat([hist_id_col, tf.expand_dims(cur_id, 1)], axis=1)
        emb_dim = int(all_ids.shape[2])

        attention_net = self.multihead_attention(all_ids, head_count, emb_dim,
                                                 seq_len, seq_size)
        tmp_net = self.add_and_norm(all_ids,
                                    attention_net,
                                    emb_dim,
                                    name='add_and_norm_1')
        feed_forward_net = tf.layers.dense(tmp_net,
                                           units=emb_dim,
                                           activation=tf.nn.relu,
                                           name='feed_forward_net')
        net = self.add_and_norm(tmp_net,
                                feed_forward_net,
                                emb_dim,
                                name='add_and_norm_2')
        bst_output = tf.reshape(net, [-1, seq_size * emb_dim])
        return bst_output

    def prediction(self):
        # input embeddings of user & item features
        for key in TAG_COLUMN:
            self.feature[key] = tf.strings.split(self.feature[key], '|')
            self.feature[key] = tf.sparse.slice(
                self.feature[key], [0, 0], [self.batch_size, self.seq_len - 1])

        # input layer
        with tf.variable_scope('input_layer',
                               partitioner=self.input_layer_partitioner):
            # unsequence input
            key_dict = {}
            with tf.variable_scope('unseq_input_layer', reuse=tf.AUTO_REUSE):
                unseq_emb = tf.feature_column.input_layer(
                    self.feature,
                    self.unseq_column,
                    cols_to_output_tensors=key_dict)

            # bst input
            with tf.variable_scope('bst_input_layer', reuse=tf.AUTO_REUSE):
                # tag input
                with tf.variable_scope('tag_input_layer', reuse=tf.AUTO_REUSE):
                    tag_emb, tag_len = tf.contrib.feature_column.sequence_input_layer(
                        self.feature, self.tag_column)

                # key input
                with tf.variable_scope('key_input_layer', reuse=tf.AUTO_REUSE):
                    key_emb_list = []
                    for key in self.key_column:
                        key_emb_list.append(key_dict[key])
                    key_emb = tf.concat(key_emb_list, axis=-1)

            if self.bf16:
                unseq_emb = tf.cast(unseq_emb, dtype=tf.bfloat16)
                tag_emb = tf.cast(tag_emb, dtype=tf.bfloat16)
                key_emb = tf.cast(key_emb, dtype=tf.bfloat16)

            bst_tower_fea = {
                'key': key_emb,
                'hist_seq_emb': tag_emb,
                'hist_seq_len': tag_len
            }

        # BST
        bst_scope = tf.variable_scope('bst_tower',
                                      partitioner=self.dense_layer_partitioner,
                                      reuse=tf.AUTO_REUSE)
        with bst_scope.keep_weights() if self.bf16 else bst_scope:
            tower_seq_len = self.seq_len
            tower_multi_head_size = self.multi_head_size
            bst_output = self.bst(bst_tower_fea,
                                  seq_size=tower_seq_len,
                                  head_count=tower_multi_head_size,
                                  name="bst")

        net = tf.concat([unseq_emb, bst_output], axis=1)

        # final_dnn
        final_dnn_scope = tf.variable_scope(
            'final_dnn_layer',
            partitioner=self.dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)

        with final_dnn_scope.keep_weights() if self.bf16 else final_dnn_scope:
            for idx, units in enumerate(self.final_dnn_hidden_units):
                with tf.variable_scope('final_dnn_%d' % idx,
                                       reuse=tf.AUTO_REUSE):
                    net = tf.layers.dense(net, units=units, activation=None)
                    # BN
                    if self.use_bn:
                        net = tf.layers.batch_normalization(
                            net, training=self._is_training, trainable=True)
                    # activate func
                    net = tf.nn.relu(net)

            logits = tf.layers.dense(inputs=net,
                                     units=self._num_class,
                                     kernel_regularizer=self.l2_regularization)

        if self.bf16:
            logits = tf.cast(logits, dtype=tf.float32)

        predict = tf.nn.sigmoid(logits)

        return predict

    def optimizer(self):
        loss_func = tf.keras.losses.CategoricalCrossentropy()
        self.predict = tf.squeeze(self.predict)
        loss = tf.math.reduce_mean(loss_func(self.label, self.predict))
        tf.summary.scalar('loss', loss)

        self.global_step = tf.train.get_or_create_global_step()

        # learning_rate = self.learning_rate
        min_learning_rate = 0.00001
        learning_rate = min_learning_rate + tf.train.exponential_decay(
            learning_rate=0.001,
            global_step=self.global_step,
            decay_steps=1000,
            decay_rate=0.5)
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=self.beta1,
                                           beta2=self.beta2,
                                           name='Adam')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=self.global_step)

        return train_op, loss


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
                        help='Batch size to train. Default is 4096',
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
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.001)
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
    parser.add_argument("--protocol",
                        type=str,
                        choices=["grpc", "grpc++", "star_server"],
                        default="grpc")
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner',
                        help='slice size of input layer partitioner. units MB',
                        type=int,
                        default=0)
    parser.add_argument('--dense_layer_partitioner',
                        help='slice size of dense layer partitioner. units KB',
                        type=int,
                        default=0)
    return parser


def main(tf_config=None, server=None):
    # check dataset
    print('Checking dataset')
    train_file = args.data_location + '/taobao_train_data'
    test_file = args.data_location + '/taobao_test_data'

    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print(
            '------------------------------------------------------------------------------------------'
        )
        print(
            "train.csv or eval.csv does not exist in the given data_location. Please provide valid path"
        )
        print(
            '------------------------------------------------------------------------------------------'
        )
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of training dataset is {}".format(no_of_training_examples))
    print("Numbers of test dataset is {}".format(no_of_test_examples))

    # set params
    # set batch size & steps
    batch_size = args.batch_size
    if args.steps == 0:
        no_of_epochs = 1000
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)

    # set fixed random seed
    tf.set_random_seed(2021)

    # set directory path
    model_dir = os.path.join(args.output_dir,
                             'model_BST_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline
    user_column, item_column, tag_column, key_column = build_feature_cols()
    train_dataset = generate_input_data(train_file, batch_size, no_of_epochs)
    test_dataset = generate_input_data(test_file, batch_size, 1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create variable partitioner for distributed training
    num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
    input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.input_layer_partitioner <<
        20) if args.input_layer_partitioner else None
    dense_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.dense_layer_partitioner <<
        10) if args.dense_layer_partitioner else None

    # create model
    model = BST(user_column=user_column,
                item_column=item_column,
                tag_column=tag_column,
                key_column=key_column,
                batch_size=batch_size,
                learning_rate=args.learning_rate,
                bf16=args.bf16,
                inputs=next_element,
                input_layer_partitioner=input_layer_partitioner,
                dense_layer_partitioner=dense_layer_partitioner)

    sess_config = tf.ConfigProto()
    if args.inter:
        sess_config.inter_op_parallelism_threads = args.inter
    if args.intra:
        sess_config.intra_op_parallelism_threads = args.intra
    hooks = []

    if tf_config:
        print('train steps : %d' % train_steps)
        hooks.append(tf.train.StopAtStepHook(last_step=train_steps))
        hooks.append(
            tf.train.LoggingTensorHook(
                {
                    'steps': model.global_step,
                    'loss': model.loss
                },
                every_n_iter=100))

        scaffold = tf.train.Scaffold(
            local_init_op=tf.group(tf.global_variables_initializer(
            ), tf.local_variables_initializer(), train_init_op))

        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=tf_config['is_chief'],
                checkpoint_dir=checkpoint_dir,
                scaffold=scaffold,
                hooks=hooks,
                # save_checkpoint_steps=args.save_steps,
                log_step_count_steps=100,
                config=sess_config) as sess:
            while not sess.should_stop():
                _, train_loss = sess.run([model.train_op, model.loss])
    else:
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=args.keep_checkpoint_max)
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # train model
            sess.run(train_init_op)
            model._is_training = True

            start = time.perf_counter()
            for _in in range(0, train_steps):
                if args.save_steps > 0 and (_in % args.save_steps == 0
                                            or _in == train_steps - 1):
                    _, train_loss, events = sess.run(
                        [model.train_op, model.loss, merged])
                    writer.add_summary(events, _in)
                    checkpoint_path = saver.save(sess,
                                                 save_path=os.path.join(
                                                     checkpoint_dir,
                                                     'BST-checkpoint'),
                                                 global_step=_in)
                    print("Save checkpoint to %s" % checkpoint_path)
                elif (args.timeline > 0 and _in % args.timeline == 0):
                    _, train_loss = sess.run([model.train_op, model.loss],
                                             options=options,
                                             run_metadata=run_metadata)
                    fetched_timeline = timeline.Timeline(
                        run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format(
                    )
                    print("Save timeline to %s" % checkpoint_dir)
                    with open(
                            os.path.join(checkpoint_dir,
                                         'timeline-%d.json' % _in), 'w') as f:
                        f.write(chrome_trace)
                else:
                    _, train_loss = sess.run([model.train_op, model.loss])

                # print training loss and time cost
                if (_in % 100 == 0 or _in == train_steps - 1):
                    end = time.perf_counter()
                    cost_time = end - start
                    global_step_sec = (100 if _in % 100 == 0 else train_steps -
                                       1 % 100) / cost_time
                    print("global_step/sec: %0.4f" % global_step_sec)
                    print("loss = {}, steps = {}, cost time = {:0.2f}s".format(
                        train_loss, _in, cost_time))
                    start = time.perf_counter()

            # eval model
            if not args.no_eval:
                writer = tf.summary.FileWriter(
                    os.path.join(checkpoint_dir, 'eval'))

                sess.run(test_init_op)
                sess.run(tf.local_variables_initializer())
                model.evalmodel()

                for _in in range(1, test_steps + 1):
                    if (_in != test_steps):
                        sess.run(
                            [model.acc, model.acc_op, model.auc, model.auc_op])
                        if (_in % 1000 == 0):
                            print("Evaluation complate:[{}/{}]".format(
                                _in, test_steps))
                    else:
                        _, eval_acc, _, eval_auc, events = sess.run([
                            model.acc, model.acc_op, model.auc, model.auc_op,
                            merged
                        ])
                        writer.add_summary(events, _in)
                        print("Evaluation complate:[{}/{}]".format(
                            _in, test_steps))
                        print("ACC = {}\nAUC = {}".format(eval_acc, eval_auc))


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    TF_CONFIG = os.getenv('TF_CONFIG')
    if not TF_CONFIG:
        main()
    else:
        print(TF_CONFIG)
        tf_config = json.loads(TF_CONFIG)
        cluster_config = tf_config.get('cluster')
        ps_hosts = []
        worker_hosts = []
        chief_hosts = []
        for key, value in cluster_config.items():
            if "ps" == key:
                ps_hosts = value
            elif "worker" == key:
                worker_hosts = value
            elif "chief" == key:
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
        cluster = tf.train.ClusterSpec({
            "ps": ps_hosts,
            "worker": worker_hosts
        })
        server = tf.distribute.Server(cluster,
                                      job_name=task_type,
                                      task_index=task_index,
                                      protocol=args.protocol)
        if task_type == 'ps':
            server.join()
        elif task_type == 'worker':
            with tf.device(
                    tf.train.replica_device_setter(
                        worker_device="/job:worker/task:%d" % task_index,
                        cluster=cluster)):
                main(tf_config={
                    'ps_hosts': ps_hosts,
                    'worker_hosts': worker_hosts,
                    'type': task_type,
                    'index': task_index,
                    'is_chief': is_chief
                },
                     server=server)
        else:
            print("Task type or index error.")
            sys.exit()