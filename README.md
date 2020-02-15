# name

w2v model.py  
download pre-trained fastText word vectors from (https://fasttext.cc/docs/en/crawl-vectors.html)  
load pre-trained model and convert it to gensim fastText model.  
retrain the model using training data.  

`
    ft.build_vocab(data, update=True)  
    ft.train(data, total_examples=ft.corpus_count,total_words=ft.corpus_total_words, epochs=ft.epochs)  
    ft.save('w2v FastText +wiki_en d=300.bin')
`

sif
