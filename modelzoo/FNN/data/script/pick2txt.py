import pickle

def pkl2txt(filename):
    pklfile = pickle.load(open(filename+'.pkl', 'rb'))
    with open(filename+'.txt','w') as f:
        f.write('\n'.join(pklfile))




if __name__ == '__main__':
    pkl2txt('uid_voc')
    pkl2txt('mid_voc')
    pkl2txt('cat_voc')