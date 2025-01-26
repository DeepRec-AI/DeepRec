import random

NEG_SEQ_LENGTH_FOR_EACH_HISTORY_ITEM = 1


def createNegData(file):
    with open(file, 'r') as f_raw:
        with open(file + '_neg', 'w') as f_out:
            FirstLine = True
            for line in f_raw:
                linelist = line.strip().split('\t')
                uid = linelist[1]

                if uid not in user_history_behavior:
                    str = '\t'
                else:
                    his_items = linelist[4].split('')
                    neg_items_str = ''
                    neg_cates_str = ''
                    for pos in his_items:
                        tmp_items_str = ''
                        tmp_cates_str = ''
                        tmp_items = []
                        tmp_cates = []
                        neg_length = 0
                        while (True):
                            index = random.randint(
                                0,
                                len(user_history_behavior[uid][0]) - 1)
                            if user_history_behavior[uid][0][index] != pos:
                                tmp_items.append(
                                    user_history_behavior[uid][0][index])
                                tmp_cates.append(
                                    user_history_behavior[uid][1][index])
                                neg_length += 1
                            if neg_length >= NEG_SEQ_LENGTH_FOR_EACH_HISTORY_ITEM:
                                break
                        for item in tmp_items:
                            tmp_items_str += (item + '')
                        for cate in tmp_cates:
                            tmp_cates_str += (cate + '')
                        neg_items_str += (tmp_items_str[:-1] + '')
                        neg_cates_str += (tmp_cates_str[:-1] + '')
                    str = neg_items_str[:-1] + '\t' + neg_cates_str[:-1]
                if FirstLine:
                    f_out.write(str)
                    FirstLine = False
                else:
                    f_out.write('\n' + str)


user_history_behavior = {}
with open('user_history_behavior.txt', 'r') as f:
    for line in f:
        linelist = line.strip().split('\t')
        uid = linelist[0]
        items = linelist[1].split('')
        cates = linelist[2].split('')
        user_history_behavior[uid] = [items, cates]

data_file = ['local_test_splitByUser', 'local_train_splitByUser']
for file in data_file:
    createNegData(file)
