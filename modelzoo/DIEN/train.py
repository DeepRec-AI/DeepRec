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

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column import utils as fc_utils

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

UNSEQ_COLUMNS = ['UID', 'ITEM', 'CATEGORY']
HIS_COLUMNS = ['HISTORY_ITEM', 'HISTORY_CATEGORY']
NEG_COLUMNS = ['NOCLK_HISTORY_ITEM', 'NOCLK_HISTORY_CATEGORY']
SEQ_COLUMNS = HIS_COLUMNS + NEG_COLUMNS
LABEL_COLUMN = ["CLICKED"]
TRAIN_DATA_COLUMNS = LABEL_COLUMN + UNSEQ_COLUMNS + SEQ_COLUMNS

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
MAX_SEQ_LENGTH = 50


def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                      tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def _assert_all_equal_and_return(tensors, name=None):
    """Asserts that all tensors are equal and returns the first one."""
    with tf.name_scope(name, 'assert_all_equal', values=tensors):
        if len(tensors) == 1:
            return tensors[0]
        assert_equal_ops = []
        for t in tensors[1:]:
            assert_equal_ops.append(tf.debugging.assert_equal(tensors[0], t))
        with tf.control_dependencies(assert_equal_ops):
            return tf.identity(tensors[0])


def generate_input_data(filename, batch_size, num_epochs):
    def parse_csv(value, neg_value):
        tf.logging.info('Parsing {}'.format(filename))
        cate_defaults = [[" "] for i in range(0, 5)]
        # cate_defaults = [[" "] for i in range(0, 7)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cate_defaults
        columns = tf.io.decode_csv(value,
                                   record_defaults=record_defaults,
                                   field_delim='\t')
        neg_columns = tf.io.decode_csv(neg_value,
                                       record_defaults=[[""], [""]],
                                       field_delim='\t')
        columns.extend(neg_columns)
        all_columns = collections.OrderedDict(zip(column_headers, columns))

        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(filename)
    dataset_neg_samples = tf.data.TextLineDataset(filename + '_neg')
    dataset = tf.data.Dataset.zip((dataset, dataset_neg_samples))
    dataset = dataset.shuffle(buffer_size=20000,
                              seed=2021)  # set seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(1)
    return dataset


def build_feature_cols(data_location=None):
    # uid_file
    uid_file = os.path.join(data_location, 'uid_voc.txt')
    mid_file = os.path.join(data_location, 'mid_voc.txt')
    cat_file = os.path.join(data_location, 'cat_voc.txt')
    if (not os.path.exists(uid_file)) or (not os.path.exists(mid_file)) or (
            not os.path.exists(cat_file)):
        print(
            "uid_voc.txt, mid_voc.txt or cat_voc does not exist in data file.")
        sys.exit()
    # uid
    uid_cate_column = tf.feature_column.categorical_column_with_vocabulary_file(
        'UID', uid_file, default_value=0)
    uid_emb_column = tf.feature_column.embedding_column(
        uid_cate_column, dimension=EMBEDDING_DIM)

    # item
    item_cate_column = tf.feature_column.categorical_column_with_vocabulary_file(
        'ITEM', mid_file, default_value=0)
    category_cate_column = tf.feature_column.categorical_column_with_vocabulary_file(
        'CATEGORY', cat_file, default_value=0)

    # history behavior
    his_item_cate_column = tf.feature_column.sequence_categorical_column_with_vocabulary_file(
        'HISTORY_ITEM', mid_file, default_value=0)
    his_category_cate_column = tf.feature_column.sequence_categorical_column_with_vocabulary_file(
        'HISTORY_CATEGORY', cat_file, default_value=0)

    # negative samples
    noclk_his_item_cate_column = tf.feature_column.sequence_categorical_column_with_vocabulary_file(
        'NOCLK_HISTORY_ITEM', mid_file, default_value=0)
    noclk_his_category_cate_column = tf.feature_column.sequence_categorical_column_with_vocabulary_file(
        'NOCLK_HISTORY_CATEGORY', cat_file, default_value=0)

    return {
        'uid_emb_column': uid_emb_column,
        'item_cate_column': item_cate_column,
        'category_cate_column': category_cate_column,
        'his_item_cate_column': his_item_cate_column,
        'his_category_cate_column': his_category_cate_column,
        'noclk_his_item_cate_column': noclk_his_item_cate_column,
        'noclk_his_category_cate_column': noclk_his_category_cate_column
    }


class VecAttGRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.math.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state):
        return self.call(inputs, state)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        _inputs = inputs[0]
        att_score = inputs[1]
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = tf.constant_initializer(1.0, dtype=_inputs.dtype)
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [_inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = tf.math.sigmoid(self._gate_linear([_inputs, state]))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with tf.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [_inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([_inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


class DIEN():
    def __init__(self,
                 feature_column=None,
                 learning_rate=0.001,
                 embedding_dim=18,
                 hidden_size=36,
                 attention_size=36,
                 inputs=None,
                 bf16=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        if not feature_column:
            raise ValueError('Dense column or sparse column is not defined.')
        # self.feature_column = feature_column
        self.uid_emb_column = feature_column['uid_emb_column']
        self.item_cate_column = feature_column['item_cate_column']
        self.his_item_cate_column = feature_column['his_item_cate_column']
        self.category_cate_column = feature_column['category_cate_column']
        self.his_category_cate_column = feature_column[
            'his_category_cate_column']
        self.noclk_his_item_cate_column = feature_column[
            'noclk_his_item_cate_column']
        self.noclk_his_category_cate_column = feature_column[
            'noclk_his_category_cate_column']

        self.learning_rate = learning_rate
        self.input_layer_partitioner = input_layer_partitioner
        self.dense_layer_partitioner = dense_layer_partitioner

        self.feature = inputs[0]
        self.label = inputs[1]
        self.batch_size = tf.shape(self.label)[0]
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.bf16 = bf16
        if self.bf16:
            self.data_tpye = tf.bfloat16
        else:
            self.data_tpye = tf.float32

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

    def prelu(self, x, scope=''):
        """parametric ReLU activation"""
        with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
            alpha = tf.get_variable("prelu_" + scope,
                                    shape=x.get_shape()[-1],
                                    dtype=x.dtype,
                                    initializer=tf.constant_initializer(0.1))
            pos = tf.nn.relu(x)
            neg = alpha * (x - abs(x)) * tf.constant(0.5, dtype=x.dtype)
            return pos + neg

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_,
                                            name='bn1' + stag,
                                            reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1,
                               100,
                               activation=None,
                               name='f1' + stag,
                               reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1,
                               50,
                               activation=None,
                               name='f2' + stag,
                               reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2,
                               2,
                               activation=None,
                               name='f3' + stag,
                               reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + tf.constant(0.00000001, dtype=dnn3.dtype)
        return y_hat

    def auxiliary_loss(self,
                       h_states,
                       click_seq,
                       noclick_seq,
                       mask,
                       dtype=tf.float32,
                       stag=None):
        mask = tf.cast(mask, dtype=dtype)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        if dtype == tf.bfloat16:
            with tf.variable_scope('auxiliary_net').keep_weights(dtype=tf.float32):
                click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :,
                                                                          0]
                noclick_prop_ = self.auxiliary_net(noclick_input_,
                                                   stag=stag)[:, :, 0]
        else:

            with tf.variable_scope('auxiliary_net'):
                click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :,
                                                                          0]
                noclick_prop_ = self.auxiliary_net(noclick_input_,
                                                   stag=stag)[:, :, 0]

        click_loss_ = -tf.reshape(tf.log(click_prop_),
                                  [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = -tf.reshape(tf.log(1.0 - noclick_prop_),
                                    [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def attention(self,
                  query,
                  facts,
                  attention_size,
                  mask,
                  stag='null',
                  mode='SUM',
                  softmax_stag=1,
                  time_major=False,
                  return_alphas=False,
                  forCnn=False):
        if isinstance(facts, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            facts = tf.concat(facts, 2)
        if len(facts.get_shape().as_list()) == 2:
            facts = tf.expand_dims(facts, 1)

        if time_major:
            # (T,B,D) => (B,T,D)
            facts = tf.array_ops.transpose(facts, [1, 0, 2])
        # Trainable parameters
        # mask = tf.equal(mask, tf.ones_like(mask))
        facts_size = facts.get_shape().as_list()[
            -1]  # D value - hidden size of the RNN layer
        querry_size = query.get_shape().as_list()[-1]
        query = tf.layers.dense(query,
                                facts_size,
                                activation=None,
                                name='f1' + stag)
        query = self.prelu(query)
        queries = tf.tile(query, [1, tf.shape(facts)[1]])
        queries = tf.reshape(queries, tf.shape(facts))
        din_all = tf.concat([queries, facts, queries - facts, queries * facts],
                            axis=-1)
        d_layer_1_all = tf.layers.dense(din_all,
                                        80,
                                        activation=tf.nn.sigmoid,
                                        name='f1_att' + stag)
        d_layer_2_all = tf.layers.dense(d_layer_1_all,
                                        40,
                                        activation=tf.nn.sigmoid,
                                        name='f2_att' + stag)
        d_layer_3_all = tf.layers.dense(d_layer_2_all,
                                        1,
                                        activation=None,
                                        name='f3_att' + stag)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
        scores = d_layer_3_all
        # Mask
        # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2**32 + 1)
        if not forCnn:
            scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

        # Scale
        # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

        # Activation
        if softmax_stag:
            scores = tf.nn.softmax(scores)  # [B, 1, T]

        # Weighted sum
        if mode == 'SUM':
            output = tf.matmul(scores, facts)  # [B, 1, H]
            # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
        else:
            scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
            output = facts * tf.expand_dims(scores, -1)
            output = tf.reshape(output, tf.shape(facts))
        if return_alphas:
            return output, scores
        return output

    def dice(self, _x, axis=-1, epsilon=0.000000001, name=''):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            alphas = tf.get_variable('alpha' + name,
                                     _x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=_x.dtype)
            input_shape = list(_x.get_shape())

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[axis] = input_shape[axis]

            # case: train mode (uses stats of the current batch)
            mean = tf.reduce_mean(_x, axis=reduction_axes)
            brodcast_mean = tf.reshape(mean, broadcast_shape)
            std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon,
                                 axis=reduction_axes)
            std = tf.sqrt(std)
            brodcast_std = tf.reshape(std, broadcast_shape)
            x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
            # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
            x_p = tf.sigmoid(x_normed)

        return alphas * (1.0 - x_p) * _x + x_p * _x

    def embedding_input_layer(self,
                              builder,
                              feature_column,
                              embedding_table,
                              get_seq_len=False):
        sparse_tensors = feature_column._get_sparse_tensors(builder)
        sparse_tensors_ids = sparse_tensors.id_tensor
        sparse_tensors_weights = sparse_tensors.weight_tensor

        embedding = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=embedding_table,
            sparse_ids=sparse_tensors_ids,
            sparse_weights=sparse_tensors_weights)

        if get_seq_len:
            sequence_length = fc_utils.sequence_length_from_sparse_tensor(
                sparse_tensors_ids)
            return embedding, sequence_length
        else:
            return embedding

    def embedding_input(self):
        for key in [
                'HISTORY_ITEM', 'HISTORY_CATEGORY', 'NOCLK_HISTORY_ITEM',
                'NOCLK_HISTORY_CATEGORY'
        ]:
            self.feature[key] = tf.strings.split(self.feature[key], '')
            self.feature[key] = tf.sparse.slice(
                self.feature[key], [0, 0], [self.batch_size, MAX_SEQ_LENGTH])

        # get uid embeddings
        uid_emb = tf.feature_column.input_layer(self.feature,
                                                self.uid_emb_column)
        # get embeddings of item and category
        # create embedding table
        item_embedding_var = tf.get_variable(
            'item_embedding_var',
            [self.item_cate_column._num_buckets, self.embedding_dim])
        category_embedding_var = tf.get_variable(
            'category_embedding_var',
            [self.category_cate_column._num_buckets, self.embedding_dim])

        builder = _LazyBuilder(self.feature)
        # get item embedding concat [item_id,category]
        item_embedding = self.embedding_input_layer(builder,
                                                    self.item_cate_column,
                                                    item_embedding_var)
        category_embedding = self.embedding_input_layer(
            builder, self.category_cate_column, category_embedding_var)
        item_emb = tf.concat([item_embedding, category_embedding], 1)

        # get history item embedding concat [history_item_id,history_category] and sequence length
        his_item_embedding, his_item_sequence_length = self.embedding_input_layer(
            builder,
            self.his_item_cate_column,
            item_embedding_var,
            get_seq_len=True)
        his_category_embedding, his_category_sequence_length = self.embedding_input_layer(
            builder,
            self.his_category_cate_column,
            category_embedding_var,
            get_seq_len=True)
        sequence_lengths = [
            his_item_sequence_length, his_category_sequence_length
        ]
        his_item_emb = tf.concat([his_item_embedding, his_category_embedding],
                                 2)
        sequence_length = _assert_all_equal_and_return(sequence_lengths)

        # get negative samples item embedding
        noclk_his_item_embedding = self.embedding_input_layer(
            builder, self.noclk_his_item_cate_column, item_embedding_var)
        noclk_his_category_embedding = self.embedding_input_layer(
            builder, self.noclk_his_category_cate_column,
            category_embedding_var)

        noclk_his_item_emb = tf.concat(
            [noclk_his_item_embedding, noclk_his_category_embedding], 2)

        return uid_emb, item_emb, his_item_emb, noclk_his_item_emb, sequence_length

    def top_fc_layer(self, inputs):
        bn1 = tf.layers.batch_normalization(inputs=inputs, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='dnn1')
        if args.norelu:
            dnn1 = self.dice(dnn1, name='dice_1')
        else:
            dnn1 = tf.nn.relu(dnn1)

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='dnn2')
        if args.norelu:
            dnn2 = self.dice(dnn2, name='dice_2')
        else:
            dnn2 = tf.nn.relu(dnn2)

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='dnn3')
        logits = tf.layers.dense(dnn3, 1, activation=None, name='logits')
        add_layer_summary(dnn1, 'dnn1')
        add_layer_summary(dnn2, 'dnn2')
        add_layer_summary(dnn3, 'dnn3')
        return logits

    def prediction(self):
        # input layer to get embedding of features
        with tf.variable_scope('input_layer',
                               partitioner=self.input_layer_partitioner,
                               reuse=tf.AUTO_REUSE):
            uid_emb, item_emb, his_item_emb, noclk_his_item_emb, sequence_length = self.embedding_input(
            )
            if self.bf16:
                uid_emb = tf.cast(uid_emb, tf.bfloat16)
                item_emb = tf.cast(item_emb, tf.bfloat16)
                his_item_emb = tf.cast(his_item_emb, tf.bfloat16)
                noclk_his_item_emb = tf.cast(noclk_his_item_emb, tf.bfloat16)

            item_his_eb_sum = tf.reduce_sum(his_item_emb, 1)
            # mask = tf.sequence_mask(sequence_length, maxlen=MAX_SEQ_LENGTH)
            mask = tf.sequence_mask(sequence_length)

        # RNN layer_1
        with tf.variable_scope('rnn_1'):
            run_output_1, _ = tf.nn.dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(self.hidden_size),
                inputs=his_item_emb,
                sequence_length=sequence_length,
                dtype=self.data_tpye,
                scope="gru1")
            tf.summary.histogram('GRU_outputs', run_output_1)

        # Aux loss
        aux_loss_scope = tf.variable_scope(
            'aux_loss', partitioner=self.dense_layer_partitioner)
        with aux_loss_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else aux_loss_scope:
            self.aux_loss = self.auxiliary_loss(run_output_1[:, :-1, :],
                                                his_item_emb[:, 1:, :],
                                                noclk_his_item_emb[:, 1:, :],
                                                mask[:, 1:],
                                                dtype=self.data_tpye,
                                                stag='gru')
            if self.bf16:
                self.aux_loss = tf.cast(self.aux_loss, tf.float32)

        # Attention layer
        attention_scope = tf.variable_scope('attention_layer')
        with attention_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else attention_scope:
            _, alphas = self.attention(item_emb,
                                       run_output_1,
                                       self.attention_size,
                                       mask,
                                       softmax_stag=1,
                                       stag='1_1',
                                       mode='LIST',
                                       return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        # RNN layer_2
        with tf.variable_scope('rnn_2'):
            _, final_state2 = tf.nn.dynamic_rnn(
                VecAttGRUCell(self.hidden_size),
                inputs=[run_output_1, tf.expand_dims(alphas, -1)],
                sequence_length=sequence_length,
                dtype=self.data_tpye,
                scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        top_input = tf.concat([
            uid_emb, item_emb, item_his_eb_sum, item_emb * item_his_eb_sum,
            final_state2
        ], 1)
        # Top MLP layer
        top_mlp_scope = tf.variable_scope(
            'top_mlp_layer',
            partitioner=self.dense_layer_partitioner,
            reuse=tf.AUTO_REUSE,
        )
        with top_mlp_scope.keep_weights(dtype=tf.float32) if self.bf16 \
            else top_mlp_scope:
            self.logits = self.top_fc_layer(top_input)
        if self.bf16:
            self.logits = tf.cast(self.logits, dtype=tf.float32)

        predict = tf.math.sigmoid(self.logits)
        return predict

    def optimizer(self):
        self.logits = tf.squeeze(self.logits)
        self.crt_loss = tf.losses.sigmoid_cross_entropy(
            self.label,
            self.logits,
            scope='loss',
            reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        loss = self.crt_loss + self.aux_loss
        tf.summary.scalar('sigmoid_cross_entropy', self.crt_loss)
        tf.summary.scalar('aux_loss', self.aux_loss)

        tf.summary.scalar('loss', loss)

        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients = optimizer.compute_gradients(loss)
        clipped_gradients = [(tf.clip_by_norm(grad, 5), var)
                             for grad, var in gradients if grad is not None]

        train_op = optimizer.apply_gradients(clipped_gradients,
                                             global_step=self.global_step)

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
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=128)
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
    parser.add_argument('--norelu', action='store_true')
    return parser


def main(tf_config=None, server=None):
    # check dataset
    print('Checking dataset')
    train_file = args.data_location + '/local_train_splitByUser'
    test_file = args.data_location + '/local_test_splitByUser'

    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)) or (
            not os.path.exists(train_file + '_neg')) or (
                not os.path.exists(test_file + '_neg')):
        print("Dataset does not exist in the given data_location.")
        sys.exit()

    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of training dataset is {}".format(no_of_training_examples))
    print("Numbers of test dataset is {}".format(no_of_test_examples))

    # set params
    # set batch size & steps
    batch_size = args.batch_size
    if args.steps == 0:
        no_of_epochs = 3
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
                             'model_DIEN_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline
    feature_column = build_feature_cols(args.data_location)
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
    model = DIEN(feature_column=feature_column,
                 learning_rate=args.learning_rate,
                 embedding_dim=EMBEDDING_DIM,
                 hidden_size=HIDDEN_SIZE,
                 attention_size=ATTENTION_SIZE,
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
        hooks.append(tf.train.StopAtStepHook(last_step=train_steps))
        hooks.append(
            tf.train.LoggingTensorHook(
                {
                    'steps': model.global_step,
                    'loss': model.loss
                },
                every_n_iter=100))

        scaffold = tf.train.Scaffold(local_init_op=tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            tf.tables_initializer(),
            train_init_op))

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
            sess.run(tf.tables_initializer())
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=args.keep_checkpoint_max)
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            # train model
            sess.run(train_init_op)

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
                                                     'DIEN-checkpoint'),
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