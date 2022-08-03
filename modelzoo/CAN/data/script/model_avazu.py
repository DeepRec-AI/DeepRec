#coding:utf-8
import tensorflow as tf
from utils import *
from tensorflow.python.ops.rnn_cell import GRUCell
import mimn as mimn
import rum as rum
from rnn import dynamic_rnn 
# import mann_simple_cell as mann_cell
import random

### Exp config ###

feature_num = [
    264,7,7,4842,7912,26,9136,580,36,
    7338655,8303,5,4,2885,8,9,474,4,69,172,62
]
# id starts with 1
id_offset = [0] + [sum(feature_num[:i])  for i in range(1, len(feature_num))]

emb_as_weight = True #False #True
use_new_seq_emb = True #False # True
#edge_type = "item"
edge_type = "3-9"
use_cartes = ["item-his_item"]
use_cartes = ["cate-his_cate"]
use_cartes = [
    "3-9", "3-10", "4-9", "4-10", "6-9", "6-10", "7-9", "7-10",
    "16-9", "16-10", "19-9", "19-10", "13-16-19", "13-16-19-9", "13-16-19-10",
    "16-3", "16-6", "19-3", "19-6", "13-16-19-3", "13-16-19-6"
]
use_cartes = []

WEIGHT_EMB_NUM = 1
orders = 5
CALC_MODE = "poly_x_x4"
weight_emb_w, weight_emb_b = [], []
alpha = 1
if CALC_MODE in ["seq_sum", "seq", "emb"]:
    weight_emb_w = [[4, 3], [3,4]]
    #weight_emb_w = [[16, 3], [3,4]]
    #weight_emb_w = [[16, 3], [3,4], [4,5],[5,5]]
    weight_emb_b = [3, 0]
    #weight_emb_b = [3, 4, 5, 0]
    WEIGHT_EMB_DIM = sum([w[0]*w[1] for w in weight_emb_w]) + sum(weight_emb_b)
elif CALC_MODE.startswith("poly"):
    WEIGHT_EMB_DIM = 16 
    if "vec" in CALC_MODE:
        WEIGHT_EMB_DIM = int(WEIGHT_EMB_DIM ** 0.5)
    elif "wx_ind" in CALC_MODE:
        WEIGHT_EMB_DIM *= 2
    elif "x_ind" in CALC_MODE:
        WEIGHT_EMB_DIM *= orders
    elif "x4" in CALC_MODE:
        alpha = 4
        WEIGHT_EMB_DIM *= alpha**2

keep_fake_carte_seq = False # True
carte_with_gru = True #False

carte_num_dict = {
    "3-6": 8315+1,
    "6-9": 1849306+1,
    "4-7": 4547+1,
    "3-9": 2102068+1,
    "3-10": 161045+1,
    "4-9": 2073680+1,
    "4-10": 146645+1,
    "6-9": 1851115+1,
    "6-10": 93771+1,
    "7-9": 1765776+1,
    "7-10": 23738+1,
    "16-9": 2135855+1,
    "16-10": 128321+1,
    "19-9": 1637771+1,
    "19-10": 57099+1,
    "13-16-19": 16905+1,
    "13-16-19-9": 2579867+1,
    "13-16-19-10": 447410+1,
    "16-3": 33287+1,
    "16-6": 25011+1,
    "19-3": 24748+1,
    "19-6": 22125+1,
    "13-16-19-3": 142791+1,
    "13-16-19-6": 86211+1,
}
if use_cartes:
    n_cid = sum([carte_num_dict[c] for c in use_cartes]) - (len(use_cartes) - 1)
#n_cid = 59201 #6689210 #8586832 #6689210 #6630010

def eb_as_weight(ad, his_items, dim, mode="seq"):
    ad = tf.reshape(ad, [-1, WEIGHT_EMB_DIM])
    weight, bias = [], []
    idx = 0
    for w, b in zip(weight_emb_w, weight_emb_b):
        weight.append(tf.reshape(ad[:, idx:idx+w[0]*w[1]], [-1, w[0], w[1]]))
        idx += w[0] * w[1]
        if b == 0:
            bias.append(None)
        else:
            bias.append(tf.reshape(ad[:, idx:idx+b], [-1, 1, b]))
            idx += b
 
    if mode == "seq_sum":
        his_items_sum = tf.reduce_sum(his_items, 1)
        his_items_sum = tf.reshape(his_items_sum, [-1, 1, dim])
        out_seq = tf.nn.selu(tf.matmul(his_items_sum, w_1) + b)
        out_seq = tf.matmul(out_seq, w_2)
        out = tf.reduce_sum(out_seq, 1)
    elif mode == "seq":
        his_items_ = tf.unstack(his_items, axis=1)
        out_seq = []
        for item in his_items_:
            item = tf.reshape(item, [-1, 1, dim])
            #out.append(tf.nn.selu(tf.matmul(item, w) + b))
            h = item
            for w, b in zip(weight, bias):
                h = tf.matmul(h, w)
                if b is not None:
                    h = tf.nn.selu(h + b)
            out_seq.append(h)
            #h = tf.nn.selu(tf.matmul(item, w_1) + b)
            #out_seq.append(tf.matmul(h, w_2))
        out_seq = tf.concat(out_seq, 1)
        out = tf.reduce_sum(out_seq, 1)
    elif mode == "emb":
        inp = his_items
        h = tf.reshape(inp, [-1, 1, dim])
        for w, b in zip(weight, bias):
            h = tf.matmul(h, w)
            if b is not None:
                h = tf.nn.selu(h + b)
        out = h
        out = tf.reduce_sum(out, 1)
    elif mode == "poly":
        h = tf.reshape(his_items, [-1, 1, dim])
        w = tf.reshape(ad, [-1, dim, dim])
        ww = [w**(i+1) for i in range(orders)]
        for i in range(orders):
            h = tf.matmul(h, ww[i])
            #if i < 2:
            h = tf.nn.tanh(h)
        out = h
        out = tf.reduce_sum(out, 1)
    elif mode == "poly_w":
        h = tf.reshape(his_items, [-1, 1, dim])
        w = tf.reshape(ad, [-1, dim, dim])
        ww = [w**(i+1) for i in range(orders)]
        out = []
        for i in range(orders):
            out.append(tf.nn.tanh(tf.matmul(h, ww[i])))
        out = tf.reduce_sum(tf.concat(out, axis=1), 1)
    elif mode == "poly_x":
        h = tf.reshape(his_items, [-1, 1, dim])
        w = tf.reshape(ad, [-1, dim, dim])
        hh = [h**(i+1) for i in range(orders)]
        out = []
        for i in range(orders):
            #out.append(tf.nn.tanh(tf.matmul(hh[i], w)))
            out.append(tf.matmul(hh[i], w))
        out = tf.reduce_sum(tf.concat(out, axis=1), 1)
    elif mode == "poly_x_x4":
        h = tf.reshape(his_items, [-1, 1, dim * alpha])
        w = tf.reshape(ad, [-1, dim*alpha, dim*alpha])
        hh = [h**(i+1) for i in range(orders)]
        out = []
        for i in range(orders):
            out.append(tf.nn.tanh(tf.matmul(hh[i], w)))
            #out.append(tf.matmul(hh[i], w))
        out = tf.reduce_sum(tf.concat(out, axis=1), 1)
    elif mode == "poly_x_ind":
        h = tf.reshape(his_items, [-1, 1, dim])
        ww = tf.split(ad, num_or_size_splits=orders, axis=1)
        ww = [tf.reshape(w, [-1, dim, dim]) for w in ww]
        hh = [h**(i+1) for i in range(orders)]
        out = []
        for i in range(orders):
            out.append(tf.nn.tanh(tf.matmul(hh[i], ww[i])))
            #out.append(tf.matmul(hh[i], ww[i]))
        out = tf.reduce_sum(tf.concat(out, axis=1), 1)
    elif mode == "poly_wx":
        h = tf.reshape(his_items, [-1, 1, dim])
        w = tf.reshape(ad, [-1, dim, dim])
        ww = [w**(i+1) for i in range(orders)]
        hh = [h**(i+1) for i in range(orders)]
        out = []
        for i in range(orders):
            out.append(tf.nn.tanh(tf.matmul(hh[i], w)))
            out.append(tf.nn.tanh(tf.matmul(h, ww[i])))
        out = tf.reduce_sum(tf.concat(out, axis=1), 1)
    elif mode == "poly_wx_ind":
        h = tf.reshape(his_items, [-1, 1, dim])
        ww = tf.split(ad, num_or_size_splits=2, axis=1)
        ww = [tf.reshape(w, [-1, dim, dim]) for w in ww]
        ww1 = [ww[1]**(i+1) for i in range(orders)]
        hh = [h**(i+1) for i in range(orders)]
        out = []
        for i in range(orders):
            out.append(tf.nn.tanh(tf.matmul(hh[i], ww[0])))
            out.append(tf.nn.tanh(tf.matmul(h, ww1[i])))
        out = tf.reduce_sum(tf.concat(out, axis=1), 1)
    elif mode == "poly_x_vec":
        h = tf.reshape(his_items, [-1, 1, dim])
        w = tf.reshape(ad, [-1, 1, dim])
        hh = [h**(i+1) for i in range(orders)]
        out = []
        for i in range(orders):
            out.append(tf.nn.tanh(hh[i] * w))
            #out.append(hh[i] * w)
        out = tf.reduce_sum(tf.concat(out, axis=1), 1)
        out = tf.reduce_sum(tf.concat(out, axis=1), 1)
    elif mode == "poly_pure":
        h = tf.reshape(his_items, [-1, 1, dim])
        w = tf.reshape(ad, [-1, dim, dim])
        ww = [w**(i+1) for i in range(orders)]
        hh = [h**(i+1) for i in range(orders)]
        out = []
        for i in range(orders):
            for j in range(orders):
                out.append(tf.nn.tanh(tf.matmul(hh[i], ww[j])))
        out = tf.reduce_sum(tf.concat(out, axis=1), 1)
            
    #out = tf.nn.selu(out)
    if keep_fake_carte_seq and mode=="seq":
        return out, out_seq
    return out, None

def FM(feas):
    feas = tf.stack(feas, aixs=1)
    square_of_sum = tf.reduce_sum(feas, axis=1) ** 2
    sum_of_square = tf.reduce_sum(feas ** 2, axis=1)
    return 0.5 * (square_of_sum - sum_of_square)

class Model(object):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=False, Flag="DNN"):
        self.model_flag = Flag
        self.reg = False
        self.use_negsample= use_negsample
        with tf.name_scope('Inputs'):
            self.user_batch_ph = tf.placeholder(tf.int32, [None, None], name='user_batch_ph')
            self.ad_batch_ph = tf.placeholder(tf.int32, [None, None], name='ad_batch_ph')
            self.scene_batch_ph = tf.placeholder(tf.int32, [None, None], name='scene_batch_ph')
            self.time_batch_ph = tf.placeholder(tf.int32, [None, ], name='time_batch_ph')
            self.clk_seq_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='clk_seq_batch_ph')
            self.carte_batch_ph = tf.placeholder(tf.int32, [None, None], name='carte_batch_ph')
            #self.noclk_seq_batch_ph = tf.placeholder(tf.int32, [None, None], name='noclk_seq_batch_ph')
            '''
            self.item_carte_batch_ph = tf.placeholder(tf.int32, [None, None], name='item_carte_batch_ph')
            self.cate_carte_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_carte_batch_ph')
            self.item_cate_carte_batch_ph = tf.placeholder(tf.int32, [None, None], name='item_cate_carte_batch_ph')
            self.cate_item_carte_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_item_carte_batch_ph')
            '''
            self.clk_mask = tf.placeholder(tf.float32, [None, None], name='clk_mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        # Embedding layer
        with tf.name_scope('Embedding_layer'):

            ad_ph = tf.split(self.ad_batch_ph, num_or_size_splits=10, axis=1)
            scene_ph = tf.split(self.scene_batch_ph, num_or_size_splits=6, axis=1)
            user_ph = tf.split(self.user_batch_ph, num_or_size_splits=4, axis=1)
            feature_ph = [self.time_batch_ph] + ad_ph[:2] + scene_ph + user_ph + ad_ph[2:]

            self.embedding_vars = []
            features = []
            for i, num in enumerate(feature_num):
                self.embedding_vars.append(tf.get_variable("embedding_var_fea{}".format(i), [num, EMBEDDING_DIM], trainable=True))
                features.append(tf.nn.embedding_lookup(self.embedding_vars[i], feature_ph[i] - id_offset[i]))

            self.user_batch_embedded = tf.concat(features[9:13], axis=1)
            self.ad_batch_embedded = tf.concat(features[1:3]+features[13:], axis=1)
            self.scene_batch_embedded = tf.concat(features[3:9], axis=1)
            self.time_batch_embedded = features[0]
            self.clk_seq_batch_embedded = tf.nn.embedding_lookup(self.embedding_vars[0], self.clk_seq_batch_ph)

            if use_cartes:
                self.carte_embeddings_var = [] 
                self.carte_batch_embedded = []
                for i, c in enumerate(use_cartes):
                    self.carte_embeddings_var.append(tf.get_variable("carte_embedding_var_{}".format(c), [carte_num_dict[c], EMBEDDING_DIM], trainable=True))
                    self.carte_batch_embedded.append(tf.nn.embedding_lookup(self.carte_embeddings_var[i], self.carte_batch_ph[:, i]))

            ###  fake carte ###
            if emb_as_weight:
                '''
                TODO: support multi-group cartesian feature, e.g., 13-16-19
                '''
                idx_w, idx_x = map(int, edge_type.split('-'))
 
                self.weight_embeddings_var = tf.get_variable("weight_embedding_var", [feature_num[idx_w] + 1, WEIGHT_EMB_NUM * WEIGHT_EMB_DIM], trainable=True)
                self.weight_batch_embedded = tf.nn.embedding_lookup(self.weight_embeddings_var, feature_ph[idx_w])
                if use_new_seq_emb:
                    self.seq_embeddings_var = tf.get_variable("seq_embedding_var", [feature_num[idx_x], EMBEDDING_DIM * alpha], trainable=True)
                    self.seq_his_batch_embedded = tf.nn.embedding_lookup(self.seq_embeddings_var, feature_ph[idx_x])

        with tf.name_scope('init_operation'):    
            for i, num in enumerate(feature_num):
                embedding_placeholder = tf.placeholder(tf.float32,[num, EMBEDDING_DIM], name="emb_ph_{}".format(i))
                self.embedding_vars[i].assign(embedding_placeholder)

            if use_cartes:
                self.carte_embedding_placeholder = []
                self.carte_embedding_init = []
                for i, c in enumerate(use_cartes):
                    self.carte_embedding_placeholder.append(tf.placeholder(tf.float32,[carte_num_dict[c], EMBEDDING_DIM], name="cid_emb_ph"))
                    self.carte_embedding_init.append(self.carte_embeddings_var[i].assign(self.carte_embedding_placeholder[i]))

        if self.use_negsample:
            self.noclk_seq_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_seq_batch_ph')
            self.noclk_seq_batch_embedded = tf.nn.embedding_lookup(self.embeddings_var, self.noclk_seq_batch_ph)
            self.noclk_mask = tf.placeholder(tf.float32, [None, None], name='noclk_mask_batch_ph')
            #self.mid_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_his_batch_ph')
            #self.cate_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_cate_his_batch_ph')
 
            #self.neg_item_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_neg_batch_ph)
            #self.neg_cate_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.cate_neg_batch_ph)
            #self.neg_his_eb = tf.concat([self.neg_item_his_eb,self.neg_cate_his_eb], axis=2) * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1))   
            self.noclk_seq_eb = tf.concat(tf.unstack(tf.reshape(self.noclk_seq_batch_embedded,(BATCH_SIZE, 10, SEQ_LEN, EMBEDDING_DIM)), axis=1), axis=-1)  * tf.reshape(self.noclk_mask,(BATCH_SIZE, SEQ_LEN, 1))   
            
        self.user_eb = tf.reshape(self.user_batch_embedded, [-1, EMBEDDING_DIM * 4]) # [batch, 4, dim] -> [batch, 4*dim]
        self.ad_eb = tf.reshape(self.ad_batch_embedded, [-1, EMBEDDING_DIM * 10]) 
        self.scene_eb = tf.reshape(self.scene_batch_embedded, [-1, EMBEDDING_DIM * 6]) 
        self.time_eb = self.time_batch_embedded

        self.clk_seq_eb = tf.concat(tf.unstack(tf.reshape(self.clk_seq_batch_embedded,(BATCH_SIZE, 10, SEQ_LEN, EMBEDDING_DIM)), axis=1), axis=-1) * tf.reshape(self.clk_mask, (BATCH_SIZE, SEQ_LEN, 1))
        self.clk_seq_eb_sum = tf.reduce_sum(self.clk_seq_eb, 1)


        self.carte_embs = []
        if use_cartes:
            self.carte_embs += self.carte_batch_embedded

        if emb_as_weight:
            if use_new_seq_emb:
                seq_his_batch = self.seq_his_batch_embedded
            else:
                seq_his_batch = features[int(edge_type.split('-')[1])]
            tmp_sum, tmp_seq = [], []
            if CALC_MODE.startswith("seq"):
                shape = (BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM)
            else:
                shape = (BATCH_SIZE, EMBEDDING_DIM * alpha)
            for i in range(WEIGHT_EMB_NUM):
                fake_carte_sum, fake_carte_seq = eb_as_weight(self.weight_batch_embedded[:, i * WEIGHT_EMB_DIM: (i+1) * WEIGHT_EMB_DIM], tf.reshape(seq_his_batch, shape), EMBEDDING_DIM, mode=CALC_MODE) 
                tmp_sum.append(fake_carte_sum)
                tmp_seq.append(fake_carte_seq)
            self.fake_carte_sum = tf.concat(tmp_sum, axis=1)
            if keep_fake_carte_seq:
                self.fake_carte_seq = tmp_seq
                

    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, scope='prelu_1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, scope='prelu_2')

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsample:
                self.loss += self.aux_loss
            if self.reg:
                self.loss += self.reg_loss

            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, clk_mask=None, noclk_mask = None, stag = None):
        #mask = tf.cast(mask, tf.float32)
        if noclk_mask is None:
            noclk_mask = clk_mask
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]

        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * clk_mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * noclk_mask

        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.000001
        return y_hat

    def init_uid_weight(self, sess, uid_weight):
        sess.run(self.uid_embedding_init,feed_dict={self.uid_embedding_placeholder: uid_weight})
    
    def init_mid_weight(self, sess, mid_weight):
        sess.run([self.mid_embedding_init],feed_dict={self.mid_embedding_placeholder: mid_weight})

    def save_mid_embedding_weight(self, sess):
        embedding = sess.run(self.mid_embeddings_var)
        return embedding

    def save_uid_embedding_weight(self, sess):
        embedding = sess.run(self.uid_bp_memory)
        return embedding                                 
    
    def train(self, sess, inps):
        input_dict = {
            self.user_batch_ph: inps[0],
            self.ad_batch_ph: inps[1],
            self.scene_batch_ph: inps[2],
            self.time_batch_ph: inps[3],
            self.clk_seq_batch_ph: inps[4],
            self.clk_mask: inps[6],
            self.target_ph: inps[-2],
            self.lr: inps[-1],
        }
        if use_cartes:
            input_dict[self.carte_batch_ph] = inps[-3]
        if "item-his_item" in use_cartes:
            input_dict[self.item_carte_batch_ph] = inps[10]
        if "cate-his_cate" in use_cartes:
            input_dict[self.cate_carte_batch_ph] = inps[11]
        if "item-his_cate" in use_cartes:
            input_dict[self.item_cate_carte_batch_ph] = inps[12]
        if "cate-his_item" in use_cartes:
            input_dict[self.cate_item_carte_batch_ph] = inps[13]

        if self.use_negsample:
            input_dict[self.noclk_seq_batch_ph] = inps[5]
            input_dict[self.noclk_mask] = inps[7]
            loss, aux_loss, accuracy, _ = sess.run([self.loss, self.aux_loss, self.accuracy, self.optimizer], feed_dict=input_dict)
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict=input_dict)
            aux_loss = 0
        return loss, accuracy, aux_loss            

    def calculate(self, sess, inps):
        input_dict = {
            self.user_batch_ph: inps[0],
            self.ad_batch_ph: inps[1],
            self.scene_batch_ph: inps[2],
            self.time_batch_ph: inps[3],
            self.clk_seq_batch_ph: inps[4],
            self.clk_mask: inps[6],
            self.target_ph: inps[-1],
        }
        if use_cartes:
            input_dict[self.carte_batch_ph] = inps[-2]
            
        if "item-his_item" in use_cartes:
            input_dict[self.item_carte_batch_ph] = inps[9]
        if "cate-his_cate" in use_cartes:
            input_dict[self.cate_carte_batch_ph] = inps[10]
        if "item-his_cate" in use_cartes:
            input_dict[self.item_cate_carte_batch_ph] = inps[11]
        if "cate-his_item" in use_cartes:
            input_dict[self.cate_item_carte_batch_ph] = inps[12]

        if self.use_negsample:
            input_dict[self.noclk_seq_batch_ph] = inps[5]
            input_dict[self.noclk_mask] = inps[7]
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict=input_dict)
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict=input_dict)
            aux_loss = 0
        return probs, loss, accuracy, aux_loss

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_DNN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_DNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DNN")
        
        #inp = tf.concat([self.item_eb, self.item_his_eb_sum], 1)
        if emb_as_weight:
            self.carte_embs.append(self.fake_carte_sum)
        inp = tf.concat([self.user_eb, self.ad_eb, self.scene_eb, self.time_eb] + self.carte_embs, 1)
        self.build_fcn_net(inp, use_dice=False)
 

class Model_FFM(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_DNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DNN")
        
        inp = tf.concat([self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)
        
       

class Model_PNN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_PNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="PNN")
        
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)


class Model_GRU4REC(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_GRU4REC, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="GRU4REC")
        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, final_state1 = dynamic_rnn(GRUCell(2*EMBEDDING_DIM), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
                    
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, final_state1], 1)
        self.build_fcn_net(inp, use_dice=False)
        

class Model_DIN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_DIN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="DIN")
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, HIDDEN_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.item_eb, self.item_his_eb_sum, att_fea], -1)
        self.build_fcn_net(inp, use_dice=False)


class Model_ARNN(Model):
    def __init__(self,n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_ARNN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="ARNN")
        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, final_state1 = dynamic_rnn(GRUCell(2*EMBEDDING_DIM), inputs=self.item_his_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_gru = din_attention(self.item_eb, rnn_outputs, HIDDEN_SIZE, self.mask)
            att_gru = tf.reduce_sum(att_gru, 1)

        inp = tf.concat([self.item_eb, self.item_his_eb_sum, final_state1, att_gru], -1)
        self.build_fcn_net(inp, use_dice=False)        

class Model_RUM(Model):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, MEMORY_SIZE, SEQ_LEN=400, mask_flag=True):
        super(Model_RUM, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="RUM")

        def clear_mask_state(state, begin_state, mask, t):
            state["controller_state"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1))) * begin_state["controller_state"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1)) * state["controller_state"]
            state["M"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1))) * begin_state["M"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1)) * state["M"]
            return state
      
        cell = rum.RUMCell(controller_units=HIDDEN_SIZE, memory_size=MEMORY_SIZE, memory_vector_dim=2*EMBEDDING_DIM,read_head_num=1, write_head_num=1,
            reuse=False, output_dim=HIDDEN_SIZE, clip_value=20, batch_size=BATCH_SIZE)
        
        state = cell.zero_state(BATCH_SIZE, tf.float32)
        begin_state = state
        for t in range(SEQ_LEN):
            output, state = cell(self.item_his_eb[:, t, :], state)
            if mask_flag:
                state = clear_mask_state(state, begin_state, self.mask, t)
        
        final_state = output
        before_memory = state['M']
        rum_att_hist = din_attention(self.item_eb, before_memory, HIDDEN_SIZE, None)

        inp = tf.concat([self.item_eb, self.item_his_eb_sum, final_state, tf.squeeze(rum_att_hist)], 1)

        self.build_fcn_net(inp, use_dice=False) 

class Model_DIEN(Model):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=400, use_negsample=False, use_mi_cons=False):
        super(Model_DIEN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, use_negsample, Flag="DIEN")

        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, _ = dynamic_rnn(GRUCell(10*EMBEDDING_DIM), inputs=self.clk_seq_eb,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)        
        
        if use_negsample:
            if use_mi_cons:
                #aux_loss_1 = self.info_NCE(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :], self.mask[:, 1:])
                #aux_loss_1 = self.info_NCE_aux(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :], self.neg_his_eb[:, 1:, :], self.mask[:, 1:])
                aux_loss_1 = self.mi_loss(rnn_outputs[:, :-1, :], self.clk_seq_eb[:, 1:, :],
                                             self.noclk_seq_eb[:, 1:, :], self.mask[:, 1:], stag = "mi_0")
            else:
                aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.clk_seq_eb[:, 1:, :],
                                             self.noclk_seq_eb[:, 1:, :], self.clk_mask[:, 1:], self.noclk_mask[:, 1:], stag = "bigru_0")
            self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_attention(self.ad_eb, rnn_outputs, HIDDEN_SIZE, mask=self.clk_mask, mode="LIST", return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.sequence_length, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        #inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum, self.item_eb*self.item_his_eb_sum], 1)
        #inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum, self.item_eb*self.item_his_eb_sum, self.item_carte_eb_sum], 1)
        #inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum, self.item_eb*self.item_his_eb_sum, self.cate_carte_eb_sum], 1)
        #inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum, self.item_eb*self.item_his_eb_sum, self.item_cate_carte_eb_sum], 1)
        #inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum, self.item_eb*self.item_his_eb_sum, self.cate_carte_eb_sum], 1)
        #inp = tf.concat([self.item_eb, final_state2, self.item_his_eb_sum, self.item_eb*self.item_his_eb_sum,  self.item_carte_eb_sum, self.cate_carte_eb_sum], 1)

        
        #if attention
        
        if emb_as_weight:
            if keep_fake_carte_seq:
                if carte_with_gru:
                    with tf.name_scope('rnn_3'):
                        self.fake_carte_seq, _ = dynamic_rnn(GRUCell(EMBEDDING_DIM), inputs=self.fake_carte_seq,
                                                 sequence_length=self.sequence_length, dtype=tf.float32,
                                                 scope="gru3")
 
                with tf.name_scope('Attention_layer_2'):
                    carte_att_outputs, _ = din_attention(self.mid_batch_embedded, self.fake_carte_seq, HIDDEN_SIZE, mask=self.mask, stag="carte", mode="SUM", return_alphas=True)
                self.carte_embs.append(tf.reduce_sum(carte_att_outputs, 1))
                #self.carte_embs.append(self.fake_carte_sum)
            else:
                self.carte_embs.append(self.fake_carte_sum)
        inp = tf.concat([self.user_eb, self.ad_eb, self.scene_eb, self.time_eb, final_state2, self.clk_seq_eb_sum, self.ad_eb*self.clk_seq_eb_sum] + self.carte_embs, 1)
        self.build_fcn_net(inp, use_dice=False)

    def neg_sample(self, neg_his_emb, K=10, mode="random"):
        shape = tf.shape(neg_his_emb)
        batch, seq, dim = shape[0], shape[1], shape[2]
        
        if mode == "random":
            neg = tf.expand_dims(neg_his_emb, 1) #[batch, 1, seq, dim]
            neg = tf.tile(neg, [1,seq, 1,1]) #[batch, seq, seq, dim]
            # index = tf.random_uniform((batch, seq, K), minval=0, maxval=seq, dtype=tf.int32)
            # neg = tf.batch_gather(neg, index) #[batch, seq, K, dim]
            neg = neg[:, :, :K, :]
            return neg
        elif mode == "aux":
            neg = tf.expand_dims(neg_his_emb, 1)
            return neg
            
    def mi_loss_(self, h_states, click_seq, noclick_seq, mask = None, stag = None):
        #mask = tf.cast(mask, tf.float32)
        '''
        h = self.mlp(h_states, stag = stag)
        pos = self.mlp(click_seq, stag = stag)
        neg = self.mlp(noclick_seq, stag = stag)

        scores_pos = tf.matmul(h, pos)
        scores_neg = tf.matmul(h, neg)
        joint = tf.linalg.diag_part(score_pos)
        '''
        pos = tf.concat([h_states, click_seq], axis=2)
        f_pos = self.mlp(pos) # [batch, seq, 1]

        K = 99
        neg = self.neg_sample(noclick_seq, K)
        h_states_tiled = tf.tile(tf.expand_dims(h_states, 2), [1,1,K,1]) # [batch, seq, K, dim]
        total = tf.concat([h_states_tiled, neg], axis=3)
        f_neg = self.mlp(total) #[batch, seq, K, 1]
        f_neg = tf.reduce_sum(f_neg, axis=2)
        f_total = f_pos + f_neg

        loss_ = tf.reshape(tf.log(f_pos / f_total), [-1, tf.shape(click_seq)[1]]) * mask
        loss_ = - tf.reduce_mean(loss_) 

        return loss_
    
    def mi_loss(self, h_states, click_seq, noclick_seq,  mask, stag='NCE'):
        exp = 'random_1'
        if exp == 'random_1':
            shape = tf.shape(h_states)
            batch, len_seq, dim = shape[0], shape[1], shape[2]
            Wk_ct = []
            x = tf.layers.dense(click_seq, 256, activation=None, name='pos_enc')
            x = tf.unstack(x, axis=1)
            neg = tf.layers.dense(noclick_seq, 256, activation=None, name='neg_enc')
            neg = tf.unstack(neg, axis=1)
            c_t = tf.unstack(h_states, axis=1)
            with tf.name_scope(stag):
                for i in range(len(c_t)):
                    Wk_ct.append(tf.layers.dense(c_t[i], 256, activation=None, name='W{}'.format(i)))
            #nce = 0        
            nce = []
            for i in range(len(c_t)):
                s_p = tf.reduce_sum(x[i] * Wk_ct[i], axis=1, keep_dims=True) # shape=[batch,1]
                s_n = tf.reduce_sum(neg[i] * Wk_ct[i], axis=1, keep_dims=True)
                score = tf.concat([s_p, s_n], axis=1)
                score = tf.nn.log_softmax(tf.exp(score), dim=1)
                score = tf.reshape(score[:, 0], [-1])
                nce.append(score)
            nce = tf.stack(nce, axis=1) * mask
            nce = tf.reduce_sum(nce)
            nce /= -1.0 * tf.cast(batch*len_seq, tf.float32)
            return nce
        elif exp == 'random_all':
            shape = tf.shape(h_states)
            batch, len_seq, dim = shape[0], shape[1], shape[2]
            Wk_ct = []
            x = tf.layers.dense(click_seq, 256, activation=None, name='pos_enc')
            x = tf.unstack(x, axis=1)
            neg = tf.layers.dense(noclick_seq, 256, activation=None, name='neg_enc')
            neg = tf.unstack(neg, axis=1)
            c_t = tf.unstack(h_states, axis=1)
            with tf.name_scope(stag):
                for i in range(len(c_t)):
                    Wk_ct.append(tf.layers.dense(c_t[i], 256, activation=None, name='W{}'.format(i)))
            nce = []
            for i in range(len(c_t)):
                s_p = tf.reduce_sum(x[i] * Wk_ct[i], axis=1, keep_dims=True) # shape=[batch,1]
                s_n = []
                for j in range(len(neg)):
                    s_n.append(tf.reduce_sum(neg[j] * Wk_ct[i], axis=1, keep_dims=True))
                score = tf.concat([s_p] + s_n, axis=1)
                score = tf.nn.log_softmax(tf.exp(score), dim=1)
                score = tf.reshape(score[:, 0], [-1])
                nce.append(score)
            nce = tf.stack(nce, axis=1) * mask
            nce = tf.reduce_sum(nce)
            nce /= -1.0 * tf.cast(batch*len_seq, tf.float32)
            return nce

        elif exp == 'batch_1':
            shape = tf.shape(click_seq)
            batch, len_seq, dim = shape[0], shape[1], shape[2]
            x = tf.layers.dense(click_seq, 256, activation=None, name='pos_enc')
            x = tf.unstack(x, axis=1)
            c_t = tf.unstack(h_states, axis=1)
            # different W for every step
            rand_idx = 12
            Wk_ct = []
            with tf.name_scope(stag):
                for i in range(len(c_t)):
                    Wk_ct.append(tf.layers.dense(c_t[i], 256, activation=None, name='W{}'.format(i)))
            nce = []
            for i in range(len(c_t)):
                x_i = tf.tile(x[i], [2,1])
                s_p = tf.reduce_sum(x_i[0:128, :] * Wk_ct[i], axis=1, keep_dims=True) # shape=[batch,1]
                s_n = tf.reduce_sum(x_i[rand_idx:rand_idx+128] * Wk_ct[i], axis=1, keep_dims=True) # shape=[batch,1]
                score = tf.concat([s_p, s_n], axis=1)
                score = tf.nn.log_softmax(tf.exp(score), dim=1) # softmax over batch
                score = tf.reshape(score[:, 0], [-1])
                nce.append(score)
            nce =tf.stack(nce, axis=1) * mask
            nce = tf.reduce_sum(nce)
            nce /= -1.0*tf.cast(batch*len_seq, tf.float32)
            return nce

        elif exp == 'batch_all':
            shape = tf.shape(click_seq)
            batch, len_seq, dim = shape[0], shape[1], shape[2]
            x = tf.layers.dense(click_seq, 256, activation=None, name='pos_enc')
            x = tf.unstack(x, axis=1)
            c_t = tf.unstack(h_states, axis=1)
            # different W for every step
            Wk_ct = []
            with tf.name_scope(stag):
                for i in range(len(c_t)):
                    Wk_ct.append(tf.layers.dense(c_t[i], 256, activation=None, name='W{}'.format(i)))
            nce = []
            for i in range(len(c_t)):
                score = tf.exp(tf.matmul(x[i], tf.transpose(Wk_ct[i])))
                score = tf.nn.log_softmax(score, dim=0) # softmax over batch
                nce.append(tf.linalg.diag_part(score))
                #nce += tf.reduce_sum(tf.linalg.diag_part(score))
            nce = tf.stack(nce, axis=1)  * mask
            nce = tf.reduce_sum(nce)
            nce /= -1.0*tf.cast(batch*len_seq, tf.float32)
            return nce


    def mlp(self, in_, stag='mlp'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 1024, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.layers.dense(dnn1, 512, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn3 = tf.layers.dense(dnn2, 256, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        return dnn3
        '''
        dnn4 = tf.layers.dense(dnn3, 1, activation=None, name='f4' + stag, reuse=tf.AUTO_REUSE)
        dnn4 = tf.nn.sigmoid(dnn4)
        return dnn4
        y_hat = tf.nn.softmax(dnn3) + 0.000001
        return y_hat
        '''

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, clk_mask=None, noclk_mask=None, stag=None):
        if noclk_mask is None:
            noclk_mask = clk_mask
        # postive 
        click_input = tf.concat([h_states, click_seq], -1)
        click_prop = self.auxiliary_net(click_input, stag = stag)[:, :, 0]
        click_loss = - tf.reshape(tf.log(click_prop), [-1, tf.shape(click_seq)[1]]) * clk_mask
        
        # negative
        exp = 'random_1'
        if exp =='random_1':
            return super(Model_DIEN, self).auxiliary_loss(h_states, click_seq, noclick_seq, clk_mask, noclk_mask, stag)
        elif exp == 'random_all':
            batch = 99
            noclick_seq_ = tf.tile(noclick_seq, [1,2,1]) # shape = [batch, 2 * seq, dim] for sliding window
            noclick_input = []
            for i in range(99):
                noclick_input.append(tf.concat([h_states, noclick_seq_[:, i:i+batch, :]], axis=-1))
            noclick_input = tf.concat(noclick_input, axis=0)
            mask = tf.tile(mask, [batch, 1])
        elif exp == 'batch_1':
            batch = 128
            h_states = tf.unstack(h_states, axis=1)
            click_seq = tf.unstack(click_seq, axis=1)
            noclick_input = []
            rand_idx = 12
            for i in range(len(click_seq)):
                h = h_states[i] # seq i of the batch, shape = [batch, dim]
                c = click_seq[i]
                c = tf.tile(c, [2, 1]) # sliding window
                noclick_input.append(tf.concat([h, c[rand_idx:rand_idx+batch,:]], axis=1))
            noclick_input = tf.stack(noclick_input, axis=1)
        elif exp == 'batch_all':
            batch = 128
            h_states = tf.unstack(h_states, axis=1)
            click_seq = tf.unstack(click_seq, axis=1)
            noclick_input = []
            for i in range(len(click_seq)):
                h = h_states[i] # seq i of the batch, shape = [batch, dim]
                c = click_seq[i]
                c = tf.tile(c, [2, 1]) # sliding window
                neg = []
                for i in range(1, batch):
                    neg.append(tf.concat([h, c[i:i+batch,:]], axis=1))
                noclick_input.append(tf.concat(neg, axis=0))
            noclick_input = tf.stack(noclick_input, axis=1)
            mask = tf.tile(mask, [batch-1, 1])

        noclick_prop = self.auxiliary_net(noclick_input, stag = stag)[:, :, 0]
        noclick_loss = - tf.reshape(tf.log(1.0 - noclick_prop), [-1, tf.shape(noclick_seq)[1]])  * mask
        loss_ = tf.reduce_mean(click_loss) + tf.reduce_mean(noclick_loss)
        return loss_

    def aux_batch(self, h_states, click_seq, noclick_seq, mask = None, stag = None):
        #mask = tf.cast(mask, tf.float32)
        # batch = tf.shape(h_states)[0]
        batch = 128
        click_input_ = tf.concat([h_states, click_seq], -1)
        h_states_ = tf.unstack(h_states, axis=1)
        click_seq_ = tf.unstack(click_seq, axis=1)
        neg_input_total = []
        for i in range(len(click_seq_)):
            h = h_states_[i] # seq i of the batch [batch, dim]
            c = click_seq_[i]
            c = tf.tile(c, [2, 1]) # sliding window
            neg = []
            for i in range(1, batch):
                neg.append(tf.concat([h, c[i:i+batch,:]], axis=1))
            neg_input_total.append(tf.concat(neg, axis=0))
        noclick_input_ = tf.stack(neg_input_total, axis=1)
        #noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]

        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        mask = tf.tile(mask, [batch-1, 1])
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask

        #loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        loss_ = tf.reduce_mean(click_loss_) + tf.reduce_mean(noclick_loss_)
        return loss_


       
        
class Model_MIMN(Model):
    def __init__(self, n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, MEMORY_SIZE, SEQ_LEN=400, Mem_Induction=0, Util_Reg=0, use_negsample=False, mask_flag=False):
        super(Model_MIMN, self).__init__(n_uid, n_mid, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, use_negsample, Flag="MIMN")
        self.reg = Util_Reg

        def clear_mask_state(state, begin_state, begin_channel_rnn_state, mask, cell, t):
            state["controller_state"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1))) * begin_state["controller_state"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1)) * state["controller_state"]
            state["M"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1))) * begin_state["M"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1)) * state["M"]
            state["key_M"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1))) * begin_state["key_M"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1)) * state["key_M"]
            state["sum_aggre"] = (1-tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1))) * begin_state["sum_aggre"] + tf.reshape(mask[:,t], (BATCH_SIZE, 1, 1)) * state["sum_aggre"]
            if Mem_Induction > 0:
                temp_channel_rnn_state = []
                for i in range(MEMORY_SIZE):
                    temp_channel_rnn_state.append(cell.channel_rnn_state[i] * tf.expand_dims(mask[:,t], axis=1) + begin_channel_rnn_state[i]*(1- tf.expand_dims(mask[:,t], axis=1)))
                cell.channel_rnn_state = temp_channel_rnn_state
                temp_channel_rnn_output = []
                for i in range(MEMORY_SIZE):
                    temp_output = cell.channel_rnn_output[i] * tf.expand_dims(mask[:,t], axis=1) + begin_channel_rnn_output[i]*(1- tf.expand_dims(self.mask[:,t], axis=1))
                    temp_channel_rnn_output.append(temp_output)
                cell.channel_rnn_output = temp_channel_rnn_output

            return state
      
        cell = mimn.MIMNCell(controller_units=HIDDEN_SIZE, memory_size=MEMORY_SIZE, memory_vector_dim=2*EMBEDDING_DIM,read_head_num=1, write_head_num=1,
            reuse=False, output_dim=HIDDEN_SIZE, clip_value=20, batch_size=BATCH_SIZE, mem_induction=Mem_Induction, util_reg=Util_Reg)
        
        state = cell.zero_state(BATCH_SIZE, tf.float32)
        if Mem_Induction > 0:
            begin_channel_rnn_output = cell.channel_rnn_output
        else:
            begin_channel_rnn_output = 0.0
        
        begin_state = state
        self.state_list = [state]
        self.mimn_o = []
        for t in range(SEQ_LEN):
            output, state, temp_output_list = cell(self.item_his_eb[:, t, :], state)
            if mask_flag:
                state = clear_mask_state(state, begin_state, begin_channel_rnn_output, self.mask, cell, t)
            self.mimn_o.append(output)
            self.state_list.append(state)
                
        self.mimn_o = tf.stack(self.mimn_o, axis=1)
        self.state_list.append(state)
        mean_memory = tf.reduce_mean(state['sum_aggre'], axis=-2)

        before_aggre = state['w_aggre']
        read_out, _, _ = cell(self.item_eb, state)
        
        if use_negsample:
            aux_loss_1 = self.auxiliary_loss(self.mimn_o[:, :-1, :], self.item_his_eb[:, 1:, :],
                                             self.neg_his_eb[:, 1:, :], self.mask[:, 1:], stag = "bigru_0")
            self.aux_loss = aux_loss_1  

        if self.reg:
            self.reg_loss = cell.capacity_loss(before_aggre)
        else:
            self.reg_loss = tf.zeros(1)

        if Mem_Induction == 1:
            channel_memory_tensor = tf.concat(temp_output_list, 1)            
            multi_channel_hist = din_attention(self.item_eb, channel_memory_tensor, HIDDEN_SIZE, None, stag='pal')
            inp = tf.concat([self.item_eb, self.item_his_eb_sum, read_out, tf.squeeze(multi_channel_hist), mean_memory*self.item_eb], 1)
        else:
            inp = tf.concat([self.item_eb, self.item_his_eb_sum, read_out, mean_memory*self.item_eb], 1)

        self.build_fcn_net(inp, use_dice=False) 
