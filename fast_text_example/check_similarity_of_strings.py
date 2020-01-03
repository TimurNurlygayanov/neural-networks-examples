#!/usr/bin/python3
# -*- encoding=utf8 -*-

from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath

from nltk.corpus import stopwords


# This is just full path to text file
# with many different texts and topics,
# you can set "corpus_file" equal to the path to
# your own text file:
corpus_file = datapath('lee_background.cor')

model = FT_gensim(size=100)

# build the vocabulary
model.build_vocab(corpus_file=corpus_file)

# train the model
model.train(
    corpus_file=corpus_file, epochs=model.epochs,
    total_examples=model.corpus_count, total_words=model.corpus_total_words
)

sentence1 = 'Obama speaks to the media in Illinois'.lower().split()
sentence2 = 'The president greets the press in Chicago'.lower().split()
sentence3 = 'My dog is the the best boy ever'.lower().split()
sentence4 = 'This cat hates me. She thinks she is a god'.lower().split()

stopwords = stopwords.words('english')
sentence1 = [w for w in sentence1 if w not in stopwords]
sentence2 = [w for w in sentence2 if w not in stopwords]
sentence3 = [w for w in sentence3 if w not in stopwords]
sentence4 = [w for w in sentence4 if w not in stopwords]

# If WM distance is lower, it means that sentences
# are more similar. If the distance is higher, it means
# that sentences are different.
distance = model.wv.wmdistance(sentence1, sentence2)
print(distance)

distance = model.wv.wmdistance(sentence1, sentence3)
print(distance)

distance = model.wv.wmdistance(sentence3, sentence4)
print(distance)
