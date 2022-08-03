import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from data.script.rnn import dynamic_rnn
from data.script.utils import *
from data.script.Dice import dice

#### CAN config #####
weight_emb_w = [[16, 8], [8,4]] 
weight_emb_b = [0, 0]
print(weight_emb_w, weight_emb_b)
orders = 3
order_indep = False # True
WEIGHT_EMB_DIM = (sum([w[0]*w[1] for w in weight_emb_w]) + sum(weight_emb_b)) #* orders
INDEP_NUM = 1
if order_indep:
    INDEP_NUM *= orders

print("orders: ",orders)
CALC_MODE = "can"
device = '/gpu:0'
#### CAN config #####

def gen_coaction(ad, his_items, dim, mode="can", mask=None,keep_fake_carte_seq=False):
    weight, bias = [], []
    idx = 0
    weight_orders = []
    bias_orders = []
    for i in range(orders):
        for w, b in zip(weight_emb_w, weight_emb_b):
            weight.append(tf.reshape(ad[:, idx:idx+w[0]*w[1]], [-1, w[0], w[1]]))
            idx += w[0] * w[1]
            if b == 0:
                bias.append(None)
            else:
                bias.append(tf.reshape(ad[:, idx:idx+b], [-1, 1, b]))
                idx += b
        weight_orders.append(weight)
        bias_orders.append(bias)
        if not order_indep:
            break
 
    if mode == "can":
        out_seq = []
        hh = []
        for i in range(orders):
            hh.append(his_items**(i+1))
        #hh = [sum(hh)]
        for i, h in enumerate(hh):
            if order_indep:
                weight, bias = weight_orders[i], bias_orders[i]
            else:
                weight, bias = weight_orders[0], bias_orders[0]
            for j, (w, b) in enumerate(zip(weight, bias)):
                h  = tf.matmul(h, w)
                if b is not None:
                    h = h + b
                if j != len(weight)-1:
                    h = tf.nn.tanh(h)
                out_seq.append(h)
        out_seq = tf.concat(out_seq, 2)
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1) 
            out_seq = out_seq * mask
    out = tf.reduce_sum(out_seq, 1)
    if keep_fake_carte_seq and mode=="emb":
        return out, out_seq
    return out, None

class Model(object):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False, use_softmax=True, use_coaction=False, use_cartes=False):
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.carte_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='carte_ph')
            self.lr = tf.placeholder(tf.float64, [])
            self.use_negsampling =use_negsampling
            self.use_softmax = False #use_softmax
            self.use_coaction = use_coaction
            self.use_cartes = use_cartes
            print("args:")
            print("negsampling: ", self.use_negsampling)
            print("softmax: ", self.use_softmax)
            print("co-action: ", self.use_coaction)
            print("carte: ", self.use_cartes)
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph') #generate 3 item IDs from negative sampling.
                self.noclk_cate_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cate_batch_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)

            self.cate_embeddings_var = tf.get_variable("cate_embedding_var", [n_cate, EMBEDDING_DIM])
            tf.summary.histogram('cate_embeddings_var', self.cate_embeddings_var)
            self.cate_batch_embedded = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cate_batch_ph)
            self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cate_his_batch_ph)
            if self.use_negsampling:
                self.noclk_cate_his_batch_embedded = tf.nn.embedding_lookup(self.cate_embeddings_var, self.noclk_cate_batch_ph)

            if self.use_cartes:
                self.carte_embedding_vars = []
                self.carte_batch_embedded = []
                with tf.device(device):
                    for i, num in enumerate(n_carte):
                        print("carte num:", num)
                        self.carte_embedding_vars.append(tf.get_variable("carte_embedding_var_{}".format(i), [num, EMBEDDING_DIM], trainable=True))
                        self.carte_batch_embedded.append(tf.nn.embedding_lookup(self.carte_embedding_vars[i], self.carte_batch_ph[:,i,:]))

            ###  co-action ###
            if self.use_coaction:
                ph_dict = {
                    "item": [self.mid_batch_ph, self.mid_his_batch_ph, self.mid_his_batch_embedded],
                    "cate": [self.cate_batch_ph, self.cate_his_batch_ph, self.cate_his_batch_embedded]
                }
                self.mlp_batch_embedded = []
                with tf.device(device):
                    self.item_mlp_embeddings_var = tf.get_variable("item_mlp_embedding_var", [n_mid, INDEP_NUM * WEIGHT_EMB_DIM], trainable=True)
                    self.cate_mlp_embeddings_var = tf.get_variable("cate_mlp_embedding_var", [n_cate, INDEP_NUM * WEIGHT_EMB_DIM], trainable=True)

                    self.mlp_batch_embedded.append(tf.nn.embedding_lookup(self.item_mlp_embeddings_var, ph_dict['item'][0]))
                    self.mlp_batch_embedded.append(tf.nn.embedding_lookup(self.cate_mlp_embeddings_var, ph_dict['cate'][0]))

                    self.input_batch_embedded = []
                    self.item_input_embeddings_var = tf.get_variable("item_input_embedding_var", [n_mid, weight_emb_w[0][0] * INDEP_NUM], trainable=True)
                    self.cate_input_embeddings_var = tf.get_variable("cate_input_embedding_var", [n_cate, weight_emb_w[0][0] * INDEP_NUM], trainable=True)
                    self.input_batch_embedded.append(tf.nn.embedding_lookup(self.item_input_embeddings_var, ph_dict['item'][1]))
                    self.input_batch_embedded.append(tf.nn.embedding_lookup(self.cate_input_embeddings_var, ph_dict['cate'][1]))

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cate_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cate_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cate_his_batch_embedded[:, :, 0, :]], -1)# 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                                [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 2*EMBEDDING_DIM])# cat embedding 18 concate item embedding 18.

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cate_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

        self.cross = []
        if self.use_cartes:
            if self.mask is not None:
                mask = tf.expand_dims(self.mask, axis=-1)
            for i,emb in enumerate(self.carte_batch_embedded):
                emb = emb * mask
                carte_eb_sum = tf.reduce_sum(emb, 1) 
                self.cross.append(carte_eb_sum)

        if self.use_coaction:
            input_batch = self.input_batch_embedded
            tmp_sum, tmp_seq = [], []
            if INDEP_NUM == 2:
                for i, mlp_batch in enumerate(self.mlp_batch_embedded):
                    for j, input_batch in enumerate(self.input_batch_embedded):
                        coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, WEIGHT_EMB_DIM * j:  WEIGHT_EMB_DIM * (j+1)], input_batch[:, :, weight_emb_w[0][0] * i: weight_emb_w[0][0] * (i+1)],  EMBEDDING_DIM, mode=CALC_MODE,mask=self.mask) 
                        tmp_sum.append(coaction_sum)
                        tmp_seq.append(coaction_seq)
            else:
                for i, (mlp_batch, input_batch) in enumerate(zip(self.mlp_batch_embedded, self.input_batch_embedded)):
                    coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, : INDEP_NUM * WEIGHT_EMB_DIM], input_batch[:, :, : weight_emb_w[0][0]],  EMBEDDING_DIM, mode=CALC_MODE, mask=self.mask) 
                    tmp_sum.append(coaction_sum)
                    tmp_seq.append(coaction_seq)
            
            self.coaction_sum = tf.concat(tmp_sum, axis=1)
            self.cross.append(self.coaction_sum)

    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2 if self.use_softmax else 1, activation=None, name='f3')
        return dnn3

    def build_loss(self, inp, L2=False):

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            if self.use_softmax:
                self.y_hat = tf.nn.softmax(inp) + 0.00000001
                ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            else:
                self.y_hat = tf.nn.sigmoid(inp)
                ctr_loss = - tf.reduce_mean(tf.concat([tf.log(self.y_hat + 0.00000001) * self.target_ph, tf.log(1 - self.y_hat + 0.00000001) * (1-self.target_ph)], axis=1))
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            if L2:
                self.loss += self.l2_loss

            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            if self.use_softmax:
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            else:
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)


    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2 if self.use_softmax else 1, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        if self.use_softmax:
            y_hat = tf.nn.softmax(dnn3) + 0.00000001
        else:
            y_hat = tf.nn.sigmoid(dnn3) + 0.00000001
        return y_hat


    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
                self.noclk_mid_batch_ph: inps[9],
                self.noclk_cate_batch_ph: inps[10],
                self.carte_batch_ph: inps[11]
            })
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
                self.carte_batch_ph: inps[11]
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.noclk_mid_batch_ph: inps[8],
                self.noclk_cate_batch_ph: inps[9],
                self.carte_batch_ph: inps[10]
            })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cate_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cate_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.carte_batch_ph: inps[10]
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_NCF(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=True):
        super(Model_NCF, self).__init__(n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE,
                                           use_negsampling, use_softmax)
        with tf.name_scope('ncf_embedding'):
            self.ncf_item_embedding_var = tf.get_variable("ncf_item_embedding_var", [n_mid, EMBEDDING_DIM], trainable=True)
            self.ncf_cate_embedding_var = tf.get_variable("ncf_cate_embedding_var", [n_cate, EMBEDDING_DIM], trainable=True)

            ncf_item_emb = tf.nn.embedding_lookup(self.ncf_item_embedding_var, self.mid_batch_ph)
            ncf_item_his_emb = tf.nn.embedding_lookup(self.ncf_item_embedding_var, self.mid_his_batch_ph)
            ncf_cate_emb = tf.nn.embedding_lookup(self.ncf_cate_embedding_var, self.cate_batch_ph)
            ncf_cate_his_emb = tf.nn.embedding_lookup(self.ncf_cate_embedding_var, self.cate_his_batch_ph)            

        ncf_item_his_sum = tf.reduce_mean(ncf_item_his_emb, axis=1)
        ncf_cate_his_sum = tf.reduce_mean(ncf_cate_his_emb, axis=1)
        mf = tf.concat([ncf_item_emb * ncf_item_his_sum, ncf_cate_emb * ncf_cate_his_sum], axis=1)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        logit = self.build_fcn_net(inp, mf, use_dice=False)
        self.build_loss(logit)

    def build_fcn_net(self, inp, mf, use_dice = False):
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

        dnn2 = tf.concat([dnn2, mf], axis=1)
        dnn3 = tf.layers.dense(dnn2, 2 if self.use_softmax else 1, activation=None, name='f3')
        return dnn3

def ProductLayer(feas, DIM, share=True):
    row, col = [], []
    num = len(feas)
    pair = num * (num-1) / 2
    for i in range(num - 1):
        for j in range(i+1, num):
            row.append(i)
            col.append(j)
    if share:
        p = tf.stack([feas[i] for i in row], axis=1)
        q = tf.stack([feas[i] for i in col], axis=1)
    else:
        tmp = []
        count = {}
        for i in row:
            if i not in count:
                count[i] = 0
            else:
                count[i] += 1
            k = count[i]
            tmp.append(feas[i][:, k*DIM:(k+1)*DIM])
        p = tf.stack(tmp, axis=1)
        tmp = []
        for i in col:
            if i not in count:
                count[i] = 0
            else:
                count[i] += 1
            k = count[i]
            tmp.append(feas[i][:, k*DIM:(k+1)*DIM])
        q = tf.stack(tmp, axis=1)
        
    ipnn = p * q
    ipnn = tf.reduce_sum(ipnn, axis=2, keep_dims=False)
    p = tf.expand_dims(p, axis=1)
    w = tf.get_variable("pnn_var", [DIM, pair, DIM], trainable=True)
    opnn = tf.reduce_sum((tf.multiply((tf.transpose(tf.reduce_sum(tf.multiply(p, w), axis=-1), [0, 2, 1])), q)), axis=-1)
    pnn = tf.concat([ipnn, opnn], axis=1) 
    return pnn

class Model_PNN(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=True):
        super(Model_PNN, self).__init__(n_uid, n_mid, n_cate, n_carte,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_softmax=use_softmax)
        
        fea_list = [self.mid_batch_embedded, self.cate_batch_embedded, tf.reduce_mean(self.mid_his_batch_embedded, axis=1), tf.reduce_mean(self.cate_his_batch_embedded, axis=1)]
        pnn = ProductLayer(fea_list, EMBEDDING_DIM)
        inp = tf.concat([self.uid_batch_embedded[:, :18], self.item_eb[:, :36], self.item_his_eb_sum[:, :36], pnn], 1)
        logit = self.build_fcn_net(inp, use_dice=False)
        self.build_loss(logit)

def FMLayer(feas, output_dim=1):
    feas = tf.stack(feas, axis=1)
    square_of_sum = tf.reduce_sum(feas, axis=1, keep_dims=True) ** 2
    sum_of_square = tf.reduce_sum(feas ** 2, axis=1, keep_dims=True)
    fm_term = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2, keep_dims=False)
    if output_dim==2:
        fm_term = tf.concat([fm_term, tf.zeros_like(fm_term)], axis=1)
    return fm_term

class Model_FM(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=False):
        super(Model_FM, self).__init__(n_uid, n_mid, n_cate, n_carte,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_softmax=use_softmax)
        
        w_item_var = tf.get_variable("w_item_var", [n_mid, 1], trainable=True)
        w_cate_var = tf.get_variable("w_cate_var", [n_mid, 1], trainable=True)
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.mid_batch_ph))
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.cate_batch_ph))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.mid_his_batch_ph), axis=1))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.cate_his_batch_ph), axis=1))        
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)

        wx = tf.concat(wx, axis=1)
        lr_term = tf.reduce_sum(wx, axis=1) + b

        fea_list = [self.mid_batch_embedded, self.cate_batch_embedded, tf.reduce_sum(self.mid_his_batch_embedded, axis=1), tf.reduce_sum(self.cate_his_batch_embedded, axis=1)]
        logit = tf.reduce_sum(wx, axis=1) + b + FMLayer(fea_list, 1) 

        #self.l2_loss = 2e-5 * tf.add_n([tf.nn.l2_loss(v) for v in [wx, self.item_eb, self.item_his_eb_sum]])
        self.build_loss(logit, L2=False)

class Model_FFM(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=False):
        super(Model_FFM, self).__init__(n_uid, n_mid, n_cate, n_carte,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_softmax=use_softmax)
        
        w_item_var = tf.get_variable("w_item_var", [n_mid, 1], trainable=True)
        w_cate_var = tf.get_variable("w_cate_var", [n_mid, 1], trainable=True)
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.mid_batch_ph))
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.cate_batch_ph))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.mid_his_batch_ph), axis=1))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.cate_his_batch_ph), axis=1))        
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)

        wx = tf.concat(wx, axis=1)
        lr_term = tf.reduce_sum(wx, axis=1, keep_dims=True) + b

        with tf.name_scope('FFM_embedding'):

            FFM_item_embedding_var = tf.get_variable("FFM_item_embedding_var", [n_mid, 3, EMBEDDING_DIM], trainable=True)
            FFM_cate_embedding_var = tf.get_variable("FFM_cate_embedding_var", [n_cate, 3, EMBEDDING_DIM], trainable=True)
            item_emb = tf.nn.embedding_lookup(FFM_item_embedding_var, self.mid_batch_ph)
            item_his_emb = tf.nn.embedding_lookup(FFM_item_embedding_var, self.mid_his_batch_ph)
            item_his_sum = tf.reduce_sum(item_his_emb, axis=1)

            cate_emb = tf.nn.embedding_lookup(FFM_cate_embedding_var, self.cate_batch_ph)
            cate_his_emb = tf.nn.embedding_lookup(FFM_cate_embedding_var, self.cate_his_batch_ph)            
            cate_his_sum = tf.reduce_sum(cate_his_emb, axis=1)
        
        fea_list = [item_emb, item_his_sum, cate_emb, cate_his_sum]
        feas = tf.stack(fea_list, axis=1)
        num = len(fea_list)
        rows, cols = [], []
        for i in range(num-1):
            for j in range(i+1, num):
                rows.append([i, j-1])
                cols.append([j, i])
        p = tf.transpose(tf.gather_nd(tf.transpose(feas, [1,2,0,3]), rows), [1,0,2])
        q = tf.transpose(tf.gather_nd(tf.transpose(feas, [1,2,0,3]), cols), [1,0,2])
        ffm_term = tf.reduce_sum(p * q, axis=2)
        ffm_term = tf.reduce_sum(ffm_term, axis=1, keep_dims=True)
        logit = lr_term + ffm_term
        self.build_loss(logit)


class Model_DeepFFM(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=False):
        super(Model_DeepFFM, self).__init__(n_uid, n_mid, n_cate, n_carte,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_softmax=use_softmax)
        
        w_item_var = tf.get_variable("w_item_var", [n_mid, 1], trainable=True)
        w_cate_var = tf.get_variable("w_cate_var", [n_mid, 1], trainable=True)
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.mid_batch_ph))
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.cate_batch_ph))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.mid_his_batch_ph), axis=1))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.cate_his_batch_ph), axis=1))        
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)

        wx = tf.concat(wx, axis=1)
        lr_term = tf.reduce_sum(wx, axis=1, keep_dims=True) + b

        with tf.name_scope('FFM_embedding'):

            FFM_item_embedding_var = tf.get_variable("FFM_item_embedding_var", [n_mid, 3, EMBEDDING_DIM], trainable=True)
            FFM_cate_embedding_var = tf.get_variable("FFM_cate_embedding_var", [n_cate, 3, EMBEDDING_DIM], trainable=True)
            item_emb = tf.nn.embedding_lookup(FFM_item_embedding_var, self.mid_batch_ph)
            item_his_emb = tf.nn.embedding_lookup(FFM_item_embedding_var, self.mid_his_batch_ph)
            item_his_sum = tf.reduce_sum(item_his_emb, axis=1)

            cate_emb = tf.nn.embedding_lookup(FFM_cate_embedding_var, self.cate_batch_ph)
            cate_his_emb = tf.nn.embedding_lookup(FFM_cate_embedding_var, self.cate_his_batch_ph)            
            cate_his_sum = tf.reduce_sum(cate_his_emb, axis=1)
        
        fea_list = [item_emb, item_his_sum, cate_emb, cate_his_sum]
        feas = tf.stack(fea_list, axis=1)
        num = len(fea_list)
        rows, cols = [], []
        for i in range(num-1):
            for j in range(i+1, num):
                rows.append([i, j-1])
                cols.append([j, i])
        p = tf.transpose(tf.gather_nd(tf.transpose(feas, [1,2,0,3]), rows), [1,0,2])
        q = tf.transpose(tf.gather_nd(tf.transpose(feas, [1,2,0,3]), cols), [1,0,2])
        ffm_term = tf.reduce_sum(p * q, axis=2)
        ffm_term = tf.reduce_sum(ffm_term, axis=1, keep_dims=True)
    
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        dnn_term = self.build_fcn_net(inp, use_dice=False)

        logit = dnn_term + lr_term + ffm_term
        self.build_loss(logit)

class Model_DeepFM(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=False):
        super(Model_DeepFM, self).__init__(n_uid, n_mid, n_cate, n_carte,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_softmax=use_softmax)
        w_item_var = tf.get_variable("w_item_var", [n_mid, 1], trainable=True)
        w_cate_var = tf.get_variable("w_cate_var", [n_cate, 1], trainable=True)
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.mid_batch_ph))
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.cate_batch_ph))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.mid_his_batch_ph), axis=1))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.cate_his_batch_ph), axis=1))        
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)

        wx = tf.concat(wx, axis=1)
        lr_term = tf.reduce_sum(wx, axis=1, keep_dims=True) + b

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        logit = self.build_fcn_net(inp, use_dice=False)

        fea_list = [self.mid_batch_embedded, self.cate_batch_embedded, tf.reduce_sum(self.mid_his_batch_embedded, axis=1), tf.reduce_sum(self.cate_his_batch_embedded, axis=1)]
        fm_term = FMLayer(fea_list)
        logit = tf.layers.dense(tf.concat([logit, fm_term, lr_term], axis=1), 1, activation=None, name='fm_fc')
        #self.l2_loss = 0.01 * tf.add_n([tf.nn.l2_loss(v) for v in [wx, self.item_eb, self.item_his_eb_sum]])
        self.build_loss(logit, L2=False)

def ExtremeFMLayer(feas, dim, output_dim=1):
    num = len(feas)
    feas = tf.stack(feas, axis=1) # batch, field_num, emb_dim
    hidden_nn_layers = []
    field_nums = [num]
    final_len = 0
    hidden_nn_layers.append(feas)
    final_result = []
    cross_layers = [256, 256, 256]

    split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)

    with tf.variable_scope("xfm", initializer=tf.contrib.layers.xavier_initializer(uniform=True)) as scope:
        for idx, layer_size in enumerate(cross_layers):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, field_nums[0] * field_nums[-1]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            filters = tf.get_variable(name="f_" + str(idx),
                                      shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                      dtype=tf.float32)

            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if idx != len(cross_layers) - 1:
                next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                final_len += int(layer_size / 2)
            else:
                direct_connect = curr_out
                next_hidden = 0
                final_len += layer_size
            field_nums.append(int(layer_size / 2))

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)


        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1)

        w_nn_output = tf.get_variable(name='w_nn_output',
                                      shape=[final_len, 1],
                                      dtype=tf.float32)
        b_nn_output = tf.get_variable(name='b_nn_output',
                                      shape=[1],
                                      dtype=tf.float32,
                                      initializer=tf.zeros_initializer())
        xfm_term = tf.matmul(result, w_nn_output) + b_nn_output

        if output_dim==2:
            xfm_term = tf.concat([xfm_term, tf.zeros_like(xfm_term)], axis=1)
        return xfm_term

class Model_xDeepFM(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=False):
        super(Model_xDeepFM, self).__init__(n_uid, n_mid, n_cate, n_carte,EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_softmax=use_softmax)
        
        w_item_var = tf.get_variable("w_item_var", [n_mid, 1], trainable=True)
        w_cate_var = tf.get_variable("w_cate_var", [n_cate, 1], trainable=True)
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.mid_batch_ph))
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.cate_batch_ph))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.mid_his_batch_ph), axis=1))
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.cate_his_batch_ph), axis=1))        
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)

        wx = tf.concat(wx, axis=1)
        lr_term = tf.reduce_sum(wx, axis=1, keep_dims=True) + b
   
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        mlp_term =  self.build_fcn_net(inp, use_dice=False)

        fea_list = [self.mid_batch_embedded, self.cate_batch_embedded, tf.reduce_sum(self.mid_his_batch_embedded, axis=1), tf.reduce_sum(self.cate_his_batch_embedded, axis=1)]
        fm_term = ExtremeFMLayer(fea_list, EMBEDDING_DIM)
        self.build_loss(mlp_term + fm_term)

class Model_PIN(Model):
    def __init__(self,n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN=256):
        super(Model_PIN, self).__init__(n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, 
                                           BATCH_SIZE, SEQ_LEN, Flag="PIN")
        
        inp = tf.concat([self.item_eb, self.item_his_eb_sum], 1)
        logit = self.build_fcn_net(inp, use_dice=False)

        feas = [self.mid_batch_embedded, self.cate_batch_embedded, tf.reduce_sum(self.mid_his_batch_embedded * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1)), axis=1), tf.reduce_sum(self.cate_his_batch_embedded * tf.reshape(self.mask,(BATCH_SIZE, SEQ_LEN, 1)), axis=1)]

        self.feas = feas
        row, col = [], []
        num = len(feas)
        for i in range(num - 1):
            for j in range(i+1, num):
                row.append(i)
                col.append(j)
        pairs = len(rows)
        p = tf.concat([feas[i] for i in row], axis=1)
        q = tf.concat([feas[i] for i in col], axis=1)
        pq = p * q
        inp = tf.concat([p,q,pq], axis=2) #batch, pair, 3*dim
        logit = self.pin(inp)
        self.build_loss(logit)

    def pin(self, inp):
        batch, pair, dim = inp.shape.as_list()
        with tf.variable_scope('product_network'):
            inp = tf.transpose(inp, [1,0,2])
            x = tf.layers.dense(inp, 20, activation=None, name='fc1')
            x = tf.layers.batch_normalization(x, name='bn1')
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, 1, activation=None, name='fc2')
            x = tf.layers.batch_normalization(x, name='bn2')
            x = tf.transpose(x, [1,0,2])
            sub_out = tf.reshape(x, [-1, pair * dim])

        with tf.variable_scope('network'):
            new_inp = tf.concat(self.feas+[sub_out], axis=1)
            x = tf.layers.dense(sub_out, 400, activation=tf.nn.relu, name='fc1')
            x = tf.layers.dense(x, 400, activation=tf.nn.relu, name='fc2')
            x = tf.layers.dense(x, 400, activation=tf.nn.relu, name='fc3')
            x = tf.layers.dense(x, 1, activation=None, name='fc4')
        return x

class Model_ONN(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=False):
        super(Model_ONN, self).__init__(n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_softmax=use_softmax)

        dim = 5
        self.item_embedding_var = tf.get_variable("item_embedding_var_onn", [n_mid, dim * 3], trainable=True)
        self.item_emb = tf.nn.embedding_lookup(self.item_embedding_var, self.mid_batch_ph)
        self.item_his_emb = tf.nn.embedding_lookup(self.item_embedding_var, self.mid_his_batch_ph)
        self.item_his_emb_sum = tf.reduce_mean(self.item_his_emb, axis=1)

        self.cate_embedding_var = tf.get_variable("cate_embedding_var_onn", [n_cate, dim * 3], trainable=True)
        self.cate_emb = tf.nn.embedding_lookup(self.cate_embedding_var, self.cate_batch_ph)
        self.cate_his_emb = tf.nn.embedding_lookup(self.cate_embedding_var, self.cate_his_batch_ph)            
        self.cate_his_emb_sum = tf.reduce_mean(self.cate_his_emb, axis=1)

        fea_list = [self.item_emb, self.cate_emb, self.item_his_emb_sum, self.cate_his_emb_sum]
        onn = ProductLayer(fea_list, dim, False)
        
        inp = tf.concat([self.uid_batch_embedded, self.mid_batch_embedded, self.cate_batch_embedded, tf.reduce_mean(self.mid_his_batch_embedded, axis=1), tf.reduce_mean(self.cate_his_batch_embedded, axis=1), onn], 1)
        logit = self.build_fcn_net(inp, use_dice=False)
        self.build_loss(logit)

class Model_WideDeep(Model):
    def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_WideDeep, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        # Fully connected layer
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        d_layer_wide = tf.concat([tf.concat([self.item_eb,self.item_his_eb_sum], axis=-1),
                                self.item_eb * self.item_his_eb_sum], axis=-1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=True, use_coaction=False, use_cartes=False):
        #EMBEDDING_DIM = 4
        super(Model_DNN, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                                          ATTENTION_SIZE,
                                                          use_negsampling, use_softmax=use_softmax, use_coaction=use_coaction, use_cartes=use_cartes)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum]+self.cross, 1)
        logit = self.build_fcn_net(inp, use_dice=False)
        self.build_loss(logit)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_softmax=True):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE,
                                           use_negsampling, use_softmax=use_softmax)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        logit = self.build_fcn_net(inp, use_dice=True)
        self.build_loss(logit)


class Model_DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cate, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True, use_coaction=False):
        super(Model_DIEN, self).__init__(n_uid, n_mid, n_cate, n_carte,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling, use_coaction=use_coaction)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2]+self.cross, 1)
        prop = self.build_fcn_net(inp, use_dice=True)
        self.build_loss(prop)
