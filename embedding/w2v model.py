import numpy as  np
import gensim.models
from gensim import utils
from gensim.models.fasttext import FastText
import pickle
import gensim.downloader as api


def save(filename, m):
    with open(filename, 'wb') as f:
        pickle.dump(m, f)

def fasttext():
    ft = gensim.models.FastText.load_fasttext_format('../wiki.en.bin')
    data = []
    with open('../dataProcess/vec_train', 'r', True) as f:
        for line in f.readlines():
            data.append(line.split())
    ft.build_vocab(data, update=True)
    ft.train(data, total_examples=ft.corpus_count,total_words=ft.corpus_total_words, epochs=ft.epochs)
    ft.save('w2v FastText +wiki_en d=300.bin')
    
fasttext()
