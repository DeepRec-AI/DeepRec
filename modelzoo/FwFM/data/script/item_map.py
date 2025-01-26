import sys
from tqdm import tqdm

data_file = ['local_test_splitByUser', 'local_train_splitByUser']

item_to_cate_map = {}
# 367983
for file_name in data_file:
    with open(file_name, 'r') as f:
        for line in f:
            linelist = line.strip().split('\t')
            items = linelist[4].split('')
            cates = linelist[5].split('')
            items.append(linelist[2])
            cates.append(linelist[3])
            # print(items)
            # print(cates)
            for index, item in enumerate(items):
                if item not in item_to_cate_map:
                    item_to_cate_map[item] = cates[index]

with open('item2catmap.txt', 'w') as f:
    firstline = True
    for item, cate in item_to_cate_map.items():
        if firstline:
            f.write(item + '\t' + cate)
            firstline = False
        else:
            f.write('\n' + item + '\t' + cate)
