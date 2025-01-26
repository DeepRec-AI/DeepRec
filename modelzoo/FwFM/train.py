import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from script.models.fwfm import FwFM
from script.feature_column import SparseFeat, DenseFeat, get_feature_names,VarLenSparseFeat





os.environ['CUDA_VISIBLE_DEVICES'] = '0'

UNSEQ_COLUMNS = ['UID', 'ITEM', 'CATEGORY']
LABEL_COLUMN = ['CLICKED']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + UNSEQ_COLUMNS

EMBEDDING_DIM=8

def split(x):
    key_ans = x.split(',')
    for key in key_ans:
        if key not in key2index:
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


#连续变量分箱处理
def BinMap(data,acc):
    if acc >=1 or acc<=0:
        return print('acc must less than 1 and more than 0')
    max = data.max()
    min = data.min()
    rangelist = [i+1 for i in range(int(1/acc))]
    length = len(data)-1
    data1 = data.sort_index()
    bin_res = np.array([0] * data.shape[-1], dtype=int)
    for r in rangelist:
        if r ==1:
            lower = min
        else:
            lower = data1[int(length*((r-1)*acc))]
        rank = r*acc
        i = int(length*rank)
        # x = data[np.where(data>=lower) + np.where(data<data1[i])]
        if r == rangelist[-1]:
            mask = data.loc[(data>=lower) & (data<=max)].index
        else:
            mask = data.loc[(data >= lower) & (data <data1[i])].index

        bin_res[mask]=r
    bin_res=pd.Series(bin_res,index=data.index)
    bin_res.name = data.name+'_BIN'
    return bin_res

def build_model_input(filename=None,chunkSize=1e6,loop=True):
    chunks=[]
    data = pd.read_csv(filename, encoding="utf-8", header=None, names=TRAIN_DATA_COLUMNS, iterator=True)
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop=False
    dataset = pd.concat(chunks)


    return dataset



def build_feature_columns(data_location=None):

    if data_location:
        uid_file = os.path.join(data_location, 'uid_labelencode.csv')
        mid_file = os.path.join(data_location, 'mid_labelencode.csv')
        cat_file = os.path.join(data_location, 'cat_labelencode.csv')
        if (not os.path.exists(uid_file)) or (not os.path.exists(mid_file)) or (
                    not os.path.exists(cat_file)):
            print("uid_labelencode.csv, mid_labelencode.csv or cat_labelencode.csv does not exist in data file.")
            sys.exit()

        uid_data = pd.read_csv(uid_file,encoding="utf-8")
        mid_data = pd.read_csv(mid_file,encoding="utf-8")
        cat_data = pd.read_csv(cat_file,encoding="utf-8")


        feature_column=[SparseFeat('UID', vocabulary_size=uid_data['UID'+'_encode'].max() + 1, embedding_dim=EMBEDDING_DIM,embeddings_initializer=None),
                        SparseFeat('ITEM',vocabulary_size=mid_data['ITEM'+'_encode'].max()+1,embedding_dim=EMBEDDING_DIM,embeddings_initializer=None),
                        SparseFeat('CATEGORY',vocabulary_size=cat_data['CATEGORY'+'_encode'].max()+1,embedding_dim=EMBEDDING_DIM,embeddings_initializer=None)]

    else:
        print("data_location does not exist in data file. ")
        sys.exit()


    return feature_column


def main(train_data=None,test_data=None,feature_colums=None):
    feature_names = get_feature_names(feature_colums)
    model = FwFM(feature_colums, feature_colums,task=args.task, dnn_hidden_units=args.dnn_hidden_units,l2_reg_linear=args.l2_reg_linear,
                l2_reg_embedding=args.l2_reg_embedding,l2_reg_field_strength=args.l2_reg_field_strength,l2_reg_dnn=args.l2_reg_dnn,
                seed=args.seed,dnn_dropout=args.dnn_dropout,dnn_activation=args.dnn_activation,dnn_use_bn=args.dnn_use_bn)

    if args.optimizer=='adam':
        optimizer = Adam(learning_rate=args.learning_rate, amsgrad=False)
    model.compile(optimizer, loss=args.loss,
                  metrics=args.metrics)
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if args.training:
            train_inputs = {name: train_data[name].values for name in feature_names}
            sess.run(tf.tables_initializer())
            history = model.fit(train_inputs, train_data[LABEL_COLUMN].values,
                            batch_size=args.batch_size, epochs=args.epochs,
                            verbose=args.verbose,validation_split=args.validation_split)
            saver.save(sess,args.save_path,global_step=args.save_step)

        else:
            #new_saver = tf.train.import_meta_graph(save_path+'model.ckpt.meta')

            saver.restore(sess, tf.train.latest_checkpoint(args.save_path))
            test_inputs = {name:test_data[name].values for name in feature_names}
            pred_ans = model.predict(test_inputs, batch_size=args.batch_size)


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.001)
    parser.add_argument('--save_path',
                        help='Full path to model output directory',
                        required=False,
                        default='results/')
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--training',
                        help='train or eval ',
                        type=bool,
                        default=True)
    parser.add_argument('--epochs',
                        help='Epoch to train.Default is 50',
                        type=int,
                        default=50)
    parser.add_argument('--save_step',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=1)
    parser.add_argument('--verbose',
                        help='set the random seed for tensorflow.',
                        choices=[0,1,2],
                        default=2)
    parser.add_argument('--validation_split',
                        help='Validation split.',
                        type=float,
                        default=0.2)
    parser.add_argument('--optimizer',
                        type=str, \
                        default='adam')
    parser.add_argument('--dnn_hidden_units',
                        type=tuple,
                        help='An iterable containing all the features used by deep part of the model.',
                        default=(256, 128, 64))
    parser.add_argument('--l2_reg_linear',
                        help='L2 regularizer strength applied to linear part.',
                        type=float,
                        default=0.00001)
    parser.add_argument('--l2_reg_embedding',
                        help=' L2 regularizer strength applied to embedding vector.',
                        type=float,
                        default=0.00001)
    parser.add_argument('--l2_reg_field_strength',
                        help='L2 regularizer strength applied to the field pair strength parameter.',
                        type=float,
                        default=0.00001)
    parser.add_argument('--l2_reg_dnn',
                        help='L2 regularizer strength applied to DNN.',
                        type=float,
                        default=0)
    parser.add_argument('--seed',
                        help='to use as random seed.',
                        type=int,
                        default=1024)
    parser.add_argument('--dnn_dropout',
                        help='the probability we will drop out a given DNN coordinate,float in [0,1).',
                        type=float,
                        default=0)
    parser.add_argument('--dnn_activation',
                        help='Activation function to use in DNN.',
                        type=str,
                        default='relu')
    parser.add_argument('--dnn_use_bn',
                        help='Whether use BatchNormalization before activation or not in DNN',
                        type=bool,
                        default=False)
    parser.add_argument('--task',
                        help='``"binary"`` for  binary logloss or  ``"regression"`` for regression loss.',
                        type=str,
                        choices=['binary','regression'],
                        default='binary')
    parser.add_argument('--loss',
                        type=str,
                        default='binary_crossentropy')
    parser.add_argument('--metrics',
                        type=list,
                        default=['binary_crossentropy', 'AUC'])


    return parser



if __name__=="__main__":
    path = 'dataset'
    train_path = path+'/local_train_splitByUser_to_labelencode.txt'
    test_path = path+'/local_test_splitByUser_to_labelencode.txt'
    feature_colums = build_feature_columns(path)

    train_data = build_model_input(train_path)
    test_data = build_model_input(test_path)

    feature_names = get_feature_names(feature_colums)

    parser = get_arg_parser()
    args = parser.parse_args()
    main(train_data,test_data,feature_colums)


















