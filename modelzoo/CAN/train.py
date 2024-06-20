import numpy
import pandas as pd
from data.script.data_iterator import DataIterator,prepare_data
import tensorflow as tf
from script.model import *
import time
import random
import sys
from script.utils import *
from tqdm import *
import pickle as pkl
import argparse


EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

file_location = 'data'




def eval(sess, test_data, model, model_path):

    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats, carte = prepare_data(src, tgt, return_neg=True)
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats, carte])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        #model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum



def train(train_file = file_location+"/local_train_splitByUser",
        test_file =file_location+ "/local_test_splitByUser",
        uid_voc =file_location+ "/uid_voc.pkl",
        mid_voc = file_location+"/mid_voc.pkl",
        cat_voc = file_location+"/cat_voc.pkl",
        model_type = 'CAN',
        seed = 2,
):
    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, args.batch_size, args.maxlen,
                                  shuffle_each_epoch=False, label_type=args.label_type)

        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, args.batch_size, args.maxlen,
                                 label_type=args.label_type)

        n_uid, n_mid, n_cat, n_carte = train_data.get_n()

        model = Model_DNN(n_uid, n_mid, n_cat, n_carte, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()

        count()

        lr = 0.001
        loss_sum = 0.0
        accuracy_sum = 0.
        aux_loss_sum = 0.
        for iter in range(10):
            for src, tgt in tqdm(train_data):
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats, carte = prepare_data(src, tgt, args.maxlen, return_neg=True)

                loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats, carte])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
                sys.stdout.flush()
                if (iter % 100) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' %  (iter, loss_sum / 100, accuracy_sum / 100, aux_loss_sum / 100))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % args.test_iter) == 0:
                    auc_, loss_, acc_, aux_ = eval(sess, test_data, model, best_model_path)
                    print('iter: %d --- test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % (iter, auc_, loss_, acc_, aux_))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
                if (iter % args.save_iter) == 0:
                    print('save model iter: %d' %(iter))
                    model.save(sess, model_path+"--"+str(iter))

            lr *= 0.5

def count_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))

def count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Prameter: ", total_parameters)

def test(
        train_file = file_location+"local_train_splitByUser",
        test_file = file_location+"local_test_splitByUser",
        uid_voc = file_location+"uid_voc.pkl",
        mid_voc = file_location+"mid_voc.pkl",
        cat_voc = file_location+"cat_voc.pkl",
        model_type='CAN' ,
	seed = 2
):

    model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, args.batch_size, args.maxlen)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, args.batch_size, args.maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()

        model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        model.restore(sess, model_path)
        print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f' % eval(sess, test_data, model, model_path))


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=128)
    parser.add_argument('--training',
                        help='train or test ',
                        type=bool,
                        default=True)

    parser.add_argument('--maxlen',
                        type=int,
                        default=100)

    parser.add_argument('--test_iter',
                        type=int,
                        default=8400)

    parser.add_argument('--save_iter',
                        type=int,
                        default=8400)
    parser.add_argument('--label_type',
                        type=int,
                        default=1)

    return parser



if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    if args.training:
        train()
    else:
        test()


