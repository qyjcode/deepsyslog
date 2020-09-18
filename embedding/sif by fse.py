from gensim.models import FastText
from fse.models import Average
from fse.models import SIF
from fse.models import uSIF
from fse import IndexedList

def getdata():
    data=[]
    with open('../dataProcess/HDFS prepro by gensim','r',True) as f:
        for line in f.readlines():
            data.append(line.split())
    return data

sentences = getdata()
ft = FastText.load('w2v FastText +wiki_en d=300.bin')

#model = Average(ft)
model = SIF(ft)


model.train(IndexedList(sentences))
sv=model.sv

import pickle
def save(filename, m):
    with open(filename, 'wb') as f:
        pickle.dump(m, f)
     
for i in sv.vectors:
    sentenceVector_list.append(i.tolist())

save('s2v wikien sif-fse d=300',sentenceVector_list)
