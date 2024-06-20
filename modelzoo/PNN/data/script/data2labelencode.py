import pandas as pd
import numpy as np
import pickle

UNSEQ_COLUMNS = ['UID', 'ITEM', 'CATEGORY']
HIS_COLUMNS = ['HISTORY_ITEM', 'HISTORY_CATEGORY']
SEQ_COLUMNS = HIS_COLUMNS
LABEL_COLUMN = ['CLICKED']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + UNSEQ_COLUMNS + SEQ_COLUMNS



def inputs_to_labelencode(filename):
    def encoder_dict(data, category_col):
        category_dict = data[category_col].value_counts()
        category_dict = pd.Series(np.arange(0, len(category_dict)), index=category_dict.index).to_dict()
        data[category_col + '_encode'] = data[category_col].map(category_dict).astype('int32')
        return data

    uid_file = '../CAN/data/uid_voc.txt'
    mid_file = '../CAN/data/mid_voc.txt'
    cat_file = '../CAN/data/cat_voc.txt'

    uid_data = pd.read_csv(uid_file, encoding="utf-8", header=None, names=['UID'])
    mid_data = pd.read_csv(mid_file, encoding="utf-8", header=None, names=['ITEM'])
    cat_data = pd.read_csv(cat_file, encoding="utf-8", header=None, names=['CATEGORY'])

    uid_data = encoder_dict(uid_data, 'UID')
    mid_data = encoder_dict(mid_data, 'ITEM')
    cat_data = encoder_dict(cat_data, 'CATEGORY')

    dataset = pd.read_csv(filename, encoding="utf-8",
                          header=None, names=TRAIN_DATA_COLUMNS, sep="\t", low_memory=False)
    for key in ['UID','ITEM','CATEGORY']:
        if key=='UID':
            dataset = pd.merge(dataset, uid_data, on=key, how='inner')
        elif key=='ITEM':
            dataset = pd.merge(dataset, mid_data, on=key, how='inner')
        else:
            dataset = pd.merge(dataset, cat_data, on=key, how='inner')

    dataset = dataset.drop(UNSEQ_COLUMNS + SEQ_COLUMNS, axis=1)

    dataset.to_csv(filename + '_to_labelencode.txt',index=0,header=0)
    uid_data.to_csv('dataset/uid_labelencode.csv',index=False)
    mid_data.to_csv('dataset/mid_labelencode.csv',index=False)
    cat_data.to_csv('dataset/cat_labelencode.csv',index=False)



if __name__ == '__main__':
    inputs_to_labelencode('../CAN/data/local_train_splitByUser')
    inputs_to_labelencode('../CAN/data/local_test_splitByUser')

