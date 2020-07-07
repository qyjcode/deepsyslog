# DeepSyslog

##word embedding
w2v model.py  
download pre-trained fastText word vectors from (https://fasttext.cc/docs/en/crawl-vectors.html)  
load pre-trained model and convert it to gensim fastText model.  
retrain the model using training data.  

    ft.build_vocab(data, update=True)  
    ft.train(data, total_examples=ft.corpus_count,total_words=ft.corpus_total_words, epochs=ft.epochs)  
    ft.save('w2v FastText +wiki_en d=300.bin')  

##sentence embedding
sif by fse.py
https://github.com/oborchers/Fast_Sentence_Embeddings
implement by fse
load the trained word embedding and choose SIF case of fse to generate sentence embedding

    ft = FastText.load('w2v.bin')
    model = SIF(ft)
    model.train(IndexedList(sentences))
    

