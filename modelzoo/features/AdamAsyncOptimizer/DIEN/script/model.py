import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
#from tensorflow.python.ops.rnn import dynamic_rnn
from script.rnn import dynamic_rnn
from script.utils import *
# from Dice import dice
import pandas as pd
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes


class Model(object):
    def __init__(self,
                 n_uid,
                 n_mid,
                 n_cat,
                 EMBEDDING_DIM,
                 HIDDEN_SIZE,
                 ATTENTION_SIZE,
                 use_negsampling=False):
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None],
                                                   name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None],
                                                   name='cat_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [
                None,
            ],
                                               name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [
                None,
            ],
                                               name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [
                None,
            ],
                                               name='cat_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None],
                                             name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None],
                                            name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            self.use_negsampling = use_negsampling
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(
                    tf.int32, [None, None, None], name='noclk_mid_batch_ph'
                )  #generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(
                    tf.int32, [None, None, None], name='noclk_cat_batch_ph')

        self.uid_dict = {}
        self.mid_dict = {}
        self.cat_dict = {}
        for i in range(n_uid):
            self.uid_dict[i] = 0
        for i in range(n_mid):
            self.mid_dict[i] = 0
        for i in range(n_cat):
            self.cat_dict[i] = 0
        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var",
                                                      [n_uid, EMBEDDING_DIM])
            self.uid_batch_embedded = tf.nn.embedding_lookup(
                self.uid_embeddings_var,
                math_ops.cast(self.uid_batch_ph, dtypes.int64))

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var",
                                                      [n_mid, EMBEDDING_DIM])
            self.mid_batch_embedded = tf.nn.embedding_lookup(
                self.mid_embeddings_var,
                math_ops.cast(self.mid_batch_ph, dtypes.int64))
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(
                self.mid_embeddings_var,
                math_ops.cast(self.mid_his_batch_ph, dtypes.int64))
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(
                    self.mid_embeddings_var,
                    math_ops.cast(self.noclk_mid_batch_ph, dtypes.int64))

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var",
                                                      [n_cat, EMBEDDING_DIM])
            self.cat_batch_embedded = tf.nn.embedding_lookup(
                self.cat_embeddings_var,
                math_ops.cast(self.cat_batch_ph, dtypes.int64))
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(
                self.cat_embeddings_var,
                math_ops.cast(self.cat_his_batch_ph, dtypes.int64))
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(
                    self.cat_embeddings_var,
                    math_ops.cast(self.noclk_cat_batch_ph, dtypes.int64))

        self.item_eb = tf.concat(
            [self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat(
            [self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [
                    self.noclk_mid_his_batch_embedded[:, :, 0, :],
                    self.noclk_cat_his_batch_embedded[:, :, 0, :]
                ], -1
            )  # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(
                self.noclk_item_his_eb,
                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36
                 ])  # cat embedding 18 concate item embedding 18.

            self.noclk_his_eb = tf.concat([
                self.noclk_mid_his_batch_embedded,
                self.noclk_cat_his_batch_embedded
            ], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    def build_fcn_net(self, inp, use_dice=False):
        with tf.variable_scope('top_full_connect', reuse=tf.AUTO_REUSE):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
            dnn1 = tf.layers.dense(bn1, 200, activation=None, name='dnn1')
            if use_dice:
                dnn1 = dice(dnn1, name='dice_1')
            else:
                dnn1 = prelu(dnn1, 'prelu1')

            dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='dnn2')
            if use_dice:
                dnn2 = dice(dnn2, name='dice_2')
            else:
                dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='dnn3')
            self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = -tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamAsyncOptimizer(
                learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph),
                        tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self,
                       h_states,
                       click_seq,
                       noclick_seq,
                       mask,
                       stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = -tf.reshape(tf.log(click_prop_),
                                  [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = -tf.reshape(tf.log(1.0 - noclick_prop_),
                                    [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

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
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def train(self,
              sess,
              inps,
              iter,
              summary=False,
              options=None,
              run_metadata=None):
        if summary:
            if self.use_negsampling:
                loss, accuracy, aux_loss, _, events = sess.run(
                    [
                        self.loss, self.accuracy, self.aux_loss,
                        self.optimizer, self.merged
                    ],
                    feed_dict={
                        self.uid_batch_ph: inps[0],
                        self.mid_batch_ph: inps[1],
                        self.cat_batch_ph: inps[2],
                        self.mid_his_batch_ph: inps[3],
                        self.cat_his_batch_ph: inps[4],
                        self.mask: inps[5],
                        self.target_ph: inps[6],
                        self.seq_len_ph: inps[7],
                        self.lr: inps[8],
                        self.noclk_mid_batch_ph: inps[9],
                        self.noclk_cat_batch_ph: inps[10],
                    },
                    options=options,
                    run_metadata=run_metadata)
                for val in inps[0]:
                    self.uid_dict[val] += 1
                for val in inps[1]:
                    self.mid_dict[val] += 1
                for val in inps[2]:
                    self.cat_dict[val] += 1

                return loss, accuracy, aux_loss, events
            else:
                loss, accuracy, _, events = sess.run(
                    [self.loss, self.accuracy, self.optimizer, self.merged],
                    feed_dict={
                        self.uid_batch_ph: inps[0],
                        self.mid_batch_ph: inps[1],
                        self.cat_batch_ph: inps[2],
                        self.mid_his_batch_ph: inps[3],
                        self.cat_his_batch_ph: inps[4],
                        self.mask: inps[5],
                        self.target_ph: inps[6],
                        self.seq_len_ph: inps[7],
                        self.lr: inps[8],
                    },
                    options=options,
                    run_metadata=run_metadata)
                return loss, accuracy, 0, events
        else:
            if self.use_negsampling:
                loss, accuracy, aux_loss, _ = sess.run(
                    [self.loss, self.accuracy, self.aux_loss, self.optimizer],
                    feed_dict={
                        self.uid_batch_ph: inps[0],
                        self.mid_batch_ph: inps[1],
                        self.cat_batch_ph: inps[2],
                        self.mid_his_batch_ph: inps[3],
                        self.cat_his_batch_ph: inps[4],
                        self.mask: inps[5],
                        self.target_ph: inps[6],
                        self.seq_len_ph: inps[7],
                        self.lr: inps[8],
                        self.noclk_mid_batch_ph: inps[9],
                        self.noclk_cat_batch_ph: inps[10],
                    },
                    options=options,
                    run_metadata=run_metadata)
                for val in inps[0]:
                    self.uid_dict[val] += 1
                for val in inps[1]:
                    self.mid_dict[val] += 1
                for val in inps[2]:
                    self.cat_dict[val] += 1

                return loss, accuracy, aux_loss, None
            else:
                loss, accuracy, _ = sess.run(
                    [self.loss, self.accuracy, self.optimizer],
                    feed_dict={
                        self.uid_batch_ph: inps[0],
                        self.mid_batch_ph: inps[1],
                        self.cat_batch_ph: inps[2],
                        self.mid_his_batch_ph: inps[3],
                        self.cat_his_batch_ph: inps[4],
                        self.mask: inps[5],
                        self.target_ph: inps[6],
                        self.seq_len_ph: inps[7],
                        self.lr: inps[8],
                    },
                    options=options,
                    run_metadata=run_metadata)
                return loss, accuracy, 0, None

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run(
                [self.y_hat, self.loss, self.accuracy, self.aux_loss],
                feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7],
                    self.noclk_mid_batch_ph: inps[8],
                    self.noclk_cat_batch_ph: inps[9],
                })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run(
                [self.y_hat, self.loss, self.accuracy],
                feed_dict={
                    self.uid_batch_ph: inps[0],
                    self.mid_batch_ph: inps[1],
                    self.cat_batch_ph: inps[2],
                    self.mid_his_batch_ph: inps[3],
                    self.cat_his_batch_ph: inps[4],
                    self.mask: inps[5],
                    self.target_ph: inps[6],
                    self.seq_len_ph: inps[7]
                })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def summary(self, sess, path, iter, hist):
        writer = tf.summary.FileWriter(path, sess.graph)
        writer.add_summary(hist, iter)

    def save_freq(self):
        list_uid = self.uid_dict.items()
        list_mid = self.mid_dict.items()
        list_cat = self.cat_dict.items()
        uid_k = []
        uid_v = []
        for val in list_uid:
            uid_k.append(val[0])
            uid_v.append(val[1])
        mid_k = []
        mid_v = []
        for val in list_mid:
            mid_k.append(val[0])
            mid_v.append(val[1])
        cat_k = []
        cat_v = []
        for val in list_cat:
            cat_k.append(val[0])
            cat_v.append(val[1])
        d_uid = pd.DataFrame({'key': uid_k, 'freq': uid_v})
        d_uid.to_csv("uid_freq.csv", index=False, sep=',')

        d_mid = pd.DataFrame({'key': mid_k, 'freq': mid_v})
        d_mid.to_csv("mid_freq.csv", index=False, sep=',')

        d_cat = pd.DataFrame({'key': cat_k, 'freq': cat_v})
        d_cat.to_csv("cat_freq.csv", index=False, sep=',')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class Model_DIN_V2_Gru_Vec_attGru_Neg(Model):
    def __init__(self,
                 n_uid,
                 n_mid,
                 n_cat,
                 EMBEDDING_DIM,
                 HIDDEN_SIZE,
                 ATTENTION_SIZE,
                 use_negsampling=True,
                 bf16=False):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg,
              self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                             ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE),
                                         inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph,
                                         dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        if self.use_negsampling:
            aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :],
                                             self.item_his_eb[:, 1:, :],
                                             self.noclk_item_his_eb[:, 1:, :],
                                             self.mask[:, 1:],
                                             stag="gru")
            self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb,
                                                    rnn_outputs,
                                                    ATTENTION_SIZE,
                                                    self.mask,
                                                    softmax_stag=1,
                                                    stag='1_1',
                                                    mode='LIST',
                                                    return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(
                VecAttGRUCell(HIDDEN_SIZE),
                inputs=rnn_outputs,
                att_scores=tf.expand_dims(alphas, -1),
                sequence_length=self.seq_len_ph,
                dtype=tf.float32,
                scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([
            self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
            self.item_eb * self.item_his_eb_sum, final_state2
        ], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DIN_V2_Gru_Vec_attGru_Neg_bf16(Model):
    def build_fcn_net_bf16(self, inp, use_dice=False):
        with tf.variable_scope('top_full_connect',
                               reuse=tf.AUTO_REUSE).keep_weights():
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
            dnn1 = tf.layers.dense(bn1, 200, activation=None, name='dnn1')
            if use_dice:
                dnn1 = dice(dnn1, name='dice_1')
            else:
                dnn1 = prelu(dnn1, 'prelu1')

            dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='dnn2')
            if use_dice:
                dnn2 = dice(dnn2, name='dice_2')
            else:
                dnn2 = prelu(dnn2, 'prelu2')
            dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='dnn3')
            self.y_hat = tf.nn.softmax(dnn3) + tf.cast(0.00000001,
                                                       dtype=tf.bfloat16)

        with tf.variable_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.target_ph = tf.cast(self.target_ph, dtype=tf.bfloat16)
            ctr_loss = -tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph),
                        tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss_bf16(self,
                            h_states,
                            click_seq,
                            noclick_seq,
                            mask,
                            stag=None):
        # mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = -tf.reshape(tf.log(click_prop_),
                                  [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = -tf.reshape(tf.log(1.0 - noclick_prop_),
                                    [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def __init__(self,
                 n_uid,
                 n_mid,
                 n_cat,
                 EMBEDDING_DIM,
                 HIDDEN_SIZE,
                 ATTENTION_SIZE,
                 use_negsampling=True,
                 bf16=True):
        super(Model_DIN_V2_Gru_Vec_attGru_Neg_bf16,
              self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                             ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s)
        with tf.variable_scope('rnn_1').keep_weights():
            inputs = tf.cast(self.item_his_eb, dtype=tf.bfloat16)
            # self.item_his_eb = tf.cast(self.item_his_eb, dtype=tf.bfloat16)
            rnn_outputs, _ = dynamic_rnn(
                GRUCell(HIDDEN_SIZE),
                inputs=inputs,
                # inputs=self.item_his_eb,
                sequence_length=self.seq_len_ph,
                dtype=tf.bfloat16,
                scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        self.mask = tf.cast(self.mask, dtype=tf.bfloat16)
        if self.use_negsampling:
            with tf.variable_scope('aux_loss').keep_weights():
            # with tf.variable_scope('aux_loss'):
                # rnn_outputs_fp32 = tf.cast(rnn_outputs, dtype=tf.float32)
                self.noclk_item_his_eb = tf.cast(self.noclk_item_his_eb,
                                                 dtype=tf.bfloat16)
                aux_loss_1 = self.auxiliary_loss_bf16(
                    rnn_outputs[:, :-1, :],
                    inputs[:, 1:, :],
                    self.noclk_item_his_eb[:, 1:, :],
                    self.mask[:, 1:],
                    stag="gru")
                # self.aux_loss = tf.cast(aux_loss_1, dtype=tf.float32)
                self.aux_loss = aux_loss_1

                # aux_loss_1 = self.auxiliary_loss(rnn_outputs_fp32[:, :-1, :],
                #                                  self.item_his_eb[:, 1:, :],
                #                                  self.noclk_item_his_eb[:,
                #                                                         1:, :],
                #                                  self.mask[:, 1:],
                #                                  stag="gru")
                # self.aux_loss = aux_loss_1

        # Attention layer
        with tf.variable_scope('Attention_layer_1').keep_weights():
            self.item_eb = tf.cast(self.item_eb, dtype=tf.bfloat16)
            # self.mask = tf.cast(self.mask, dtype=tf.bfloat16)
            att_outputs, alphas = din_fcn_attention(
                # tf.cast(self.item_eb, dtype=tf.bfloat16),
                self.item_eb,
                rnn_outputs,
                ATTENTION_SIZE,
                self.mask,
                # tf.cast(self.mask, dtype=tf.bfloat16),
                softmax_stag=1,
                stag='1_1',
                mode='LIST',
                return_alphas=True)

            # att_outputs, alphas = din_fcn_attention(tf.cast(self.item_eb,dtype=tf.bfloat16),
            #                                         rnn_outputs,
            #                                         ATTENTION_SIZE,
            #                                         tf.cast(self.mask,dtype=tf.bfloat16),
            #                                         softmax_stag=1,
            #                                         stag='1_1',
            #                                         mode='LIST',
            #                                         return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.variable_scope('rnn_2').keep_weights():
            # alphas = tf.cast(alphas,dtype=tf.bfloat16)
            rnn_outputs2, final_state2 = dynamic_rnn(
                VecAttGRUCell(HIDDEN_SIZE),
                inputs=rnn_outputs,
                att_scores=tf.expand_dims(alphas, -1),
                sequence_length=self.seq_len_ph,
                dtype=tf.bfloat16,
                scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)
            # final_state2 = tf.cast(final_state2, dtype=tf.float32)

        self.uid_batch_embedded = tf.cast(self.uid_batch_embedded,
                                          dtype=tf.bfloat16)
        self.item_his_eb_sum = tf.cast(self.item_his_eb_sum, dtype=tf.bfloat16)
        inp = tf.concat([
            self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
            self.item_eb * self.item_his_eb_sum, final_state2
        ], 1)
        self.build_fcn_net_bf16(inp, use_dice=True)

        # inp = tf.concat([
        #     self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
        #     self.item_eb * self.item_his_eb_sum, final_state2
        # ], 1)
        # self.build_fcn_net(inp, use_dice=True)
