import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle as pkl
import math
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,MultiLabelBinarizer
from script.models.fnn import FNN
from script.feature_column import SparseFeat, DenseFeat, get_feature_names,VarLenSparseFeat
import gc


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def split(x):
    key_ans = x.split(',')
    for key in key_ans:
        if key not in key2index:
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))

if __name__=="__main__":
    path = 'data/'
    datalist = ['1458','2259','2261','2997','3386','all']

    for file in datalist:

        data = pd.read_csv(path+file+'/train.log.txt',encoding="utf-8",
                           header=0,sep="\t",low_memory=False)

        test_data = pd.read_csv(path+file+'/test.log.txt',encoding="utf-8",
                           header=0,sep="\t",low_memory=False)


        data = data[['click','weekday','hour','useragent','IP','region', 'city', 'adexchange', 'domain', 'slotid','slotwidth',
                     'slotheight', 'slotvisibility', 'slotformat', 'creative', 'advertiser', 'slotprice']]

        test_data = test_data[['click','weekday','hour','useragent','IP','region', 'city', 'adexchange', 'domain', 'slotid','slotwidth',
                     'slotheight', 'slotvisibility', 'slotformat', 'creative', 'advertiser', 'slotprice']]

        data['istest']=0
        test_data['istest']=1
        df =  pd.concat([data, test_data], axis=0, ignore_index=True)
        del data, test_data
        gc.collect()


        df.dropna(subset=['click'],inplace=True)

        df['adexchange'].fillna(0,inplace=True)
        df['adexchange']=df['adexchange'].astype(int)


        df.fillna('unknown', inplace=True)


        dense_features = ['weekday', 'hour','region','city','adexchange','slotwidth','slotheight',
                          'advertiser', 'slotprice' ]



        sparse_features=[]

        target='click'
        for col in df.columns:
            if col not in dense_features and col not in ['istest','click']:
                lbe = LabelEncoder()
                df[col] = lbe.fit_transform(df[col])
                df[col]=lbe.fit_transform(df[col])
                sparse_features.append(col)

        mms = MinMaxScaler(feature_range=(0, 1))

        df[dense_features] = mms.fit_transform(df[dense_features])


        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].max() + 1, embedding_dim=11,embeddings_initializer=None)
                                  for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                                for feat in dense_features]

        linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = fixlen_feature_columns





        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        # 3.generate train&test input data for model
        cols = [f for f in df.columns if f not in ['click', 'istest']]
        train = df[df.istest==0][cols]
        test = df[df.istest==1][cols]

        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        gpu_options = tf.GPUOptions(allow_growth=True)


        model = FNN(linear_feature_columns, dnn_feature_columns,task='binary',dnn_hidden_units=(128, 64, 32))

        adam = Adam(learning_rate=0.001,amsgrad=False)

        model.compile(adam, "binary_crossentropy",
                      metrics=['binary_crossentropy','AUC'])

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:


            sess.run(tf.tables_initializer())
            history = model.fit(train_model_input, df[df.istest==0][target].values,
                            batch_size=128, epochs=50, verbose=2, validation_split=0.2)

            pred_ans = model.predict(test_model_input, batch_size=128)

            test_auc = roc_auc_score(df[df.istest==1][target].values,pred_ans)
            print('test_auc=',test_auc)


            with open('result/result.txt','a+') as tx:
                print(file+" test LogLoss", round(log_loss(df[df.istest==1][target].values, pred_ans), 4),file=tx)
                print(file+" test AUC", round(roc_auc_score(df[df.istest==1][target].values, pred_ans), 4),file=tx)
                print('='*50,file=tx)











