import numpy as  np
import gensim.models
from gensim import utils
from gensim.models.fasttext import FastText
import pickle
import gensim.downloader as api


def save(filename, m):
    with open(filename, 'wb') as f:
        pickle.dump(m, f)

def getdata():
    data=[]
    with open('../dataProcess/HDFS prepro by gensim','r',True) as f:
        for line in f.readlines():
            data.append(line.split())
    return data

def google():
    sentences = getdata()
    ft = FastText(sentences, min_count=1, size=300,window=5)
    print('ft ok')

    g_wv = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
    print('gw2v ok')
    print(len(ft.wv.vocab))
    print(len(g_wv.vocab))

    d={}
    for i in ft.wv.vocab:
            d[i]=g_wv[i]
    save('w2v google dict d=300',d)

    del g_wv
    old={}
    for i in ft.wv.vocab:
        old[i] = ft.wv[i]
        if i in d.keys():
            ft.wv[i]=d[i]

    ft.train(sentences,total_examples=ft.corpus_count, epochs=ft.epochs)
    #dist = np.linalg.norm(old[i] - ft.wv[i])
    ft.save('w2v FastText +google d=300.bin')

#google()

def fasttext():
    ft = gensim.models.FastText.load_fasttext_format('../wiki.en.bin')
    print('finish load',type(ft))

    data = []
    with open('../dataProcess/vec_train', 'r', True) as f:
        for line in f.readlines():
            data.append(line.split())
    print('data ok')

    ft.build_vocab(data, update=True)
    ft.train(data, total_examples=ft.corpus_count,total_words=ft.corpus_total_words, epochs=ft.epochs)
    # dist = np.linalg.norm(old[i] - ft.wv[i])
    ft.save('w2v FastText +wiki_en d=300.bin')

fasttext()
