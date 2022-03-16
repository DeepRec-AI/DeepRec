item_to_cate_map = {}
with open('item2catmap.txt', 'r') as f:
    for line in f:
        linelist = line.strip().split('\t')
        item = linelist[0]
        cate = linelist[1]
        item_to_cate_map[item] = cate

user_history_behavior = {}
with open('reviews-info', 'r') as f:
    for line in f:
        linelist = line.strip().split('\t')
        uid = linelist[0]
        item = linelist[1]
        if uid not in user_history_behavior:
            user_history_behavior[uid] = [item]
        else:
            if item not in user_history_behavior[uid]:
                user_history_behavior[uid].append(item)

FirstLine = True
with open('user_history_behavior.txt', 'w') as f:
    for uid, items in user_history_behavior.items():
        itemstr = ''
        catestr = ''
        for i in items:
            if i in item_to_cate_map:
                c = item_to_cate_map[i]
            else:
                c = 'Unknown'
            if not itemstr:
                itemstr += i
                catestr += c
            else:
                itemstr += ('' + i)
                catestr += ('' + c)
        if FirstLine:
            f.write(uid + '\t' + itemstr + '\t' + catestr)
            FirstLine = False
        else:
            f.write('\n' + uid + '\t' + itemstr + '\t' + catestr)
