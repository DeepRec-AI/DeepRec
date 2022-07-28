import os
import pandas as pd

file =  '/home/test/modelzoo/DIEN/data/local_train_splitByUser'
# if os.path.exists(file+'_neg') is True:
#     print('YES')
# else:
#     print('NOT')
data = pd.read_csv(file)
print(data.head())