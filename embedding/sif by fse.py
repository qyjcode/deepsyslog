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
event_list=[]
sv_list=[]
for i in sentences:
    if i not in event_list:
        event_list.append(i)
        print(i)
for i in sv.vectors:
    i=i.tolist()
    if i not in sv_list:
        sv_list.append(i)

print(len(sv.vectors[0]))


order=[5,22,11,9,26,6,16,18,25,3,
       2,7,10,21,13,14,27,8,15,12,
       17,23,20,28,4,19,24,29,1]

new_sentences=[]

for i in range(1,30):
    new_sentences.append(event_list [order.index(i)])
f1=open('../dataProcess/sentence format','w+',True)

for i in new_sentences:
    for j in i:
        f1.write(str(j)+' ')
    f1.write('\n')

sentenceVector_list=[]
for i in range(1,30):
    sentenceVector_list.append(sv_list[order.index(i)])

import pickle
def save(filename, m):
    with open(filename, 'wb') as f:
        pickle.dump(m, f)

save('s2v wikien sif-fse d=300',sentenceVector_list)