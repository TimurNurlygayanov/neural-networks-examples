#!/usr/bin/python3
# -*- encoding=utf8 -*-

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import gensim
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


txt = [
    'I like to eat broccoli and bananas.',
    'I munched a banana and spinach smoothie for breakfast.',
    'Chinchillas and kittens are cute.',
    'My sister adopted a kitten yesterday.',
    'Look at this cute hamster munching on a piece of broccoli.'
]
stop_words = stopwords.words('english')


def simplify(penn_tag):
    pre = penn_tag[0]
    if pre == 'J':
        return 'a'
    elif pre == 'R':
        return 'r'
    elif pre == 'V':
        return 'v'
    else:
        return 'n'


def preprocess(text):
    toks = gensim.utils.simple_preprocess(str(text), deacc=True)
    wn = WordNetLemmatizer()

    return [wn.lemmatize(tok, simplify(pos))
            for tok, pos in nltk.pos_tag(toks)
            if tok not in stop_words]


def test_eta(eta, dictionary, ntopics, print_topics=True, print_dist=True):
    np.random.seed(42) # set the random seed for repeatability

    # get the bow-format lines with the set dictionary:
    bow = [dictionary.doc2bow(line) for line in corp]

    with (np.errstate(divide='ignore')):  # ignore divide-by-zero warnings
        model = gensim.models.ldamodel.LdaModel(
            corpus=bow, id2word=dictionary, num_topics=ntopics,
            random_state=42, chunksize=100, eta=eta,
            eval_every=-1, update_every=1,
            passes=150, alpha='auto', per_word_topics=True)

    print('Perplexity: {:.2f}'.format(model.log_perplexity(bow)))
    if print_topics:
        # display the top terms for each topic
        for topic in range(ntopics):
            print('Topic {}: {}'.format(topic, [dictionary[w]
                                                for w,p in model.get_topic_terms(topic, topn=3)]))
    if print_dist:
        # display the topic probabilities for each document
        for line, bag in zip(txt, bow):
            doc_topics = ['({}, {:.1%})'.format(topic, prob)
                          for topic, prob in model.get_document_topics(bag)]
            print('{} {}'.format(line, doc_topics))

    return model


def create_eta(priors, etadict, ntopics):
    # create a (ntopics, nterms) matrix and fill with 1
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1)

    for word, topic in priors.items():
        # look up the word in the dictionary
        keyindex = [index for index, term in etadict.items() if term == word]

        # if it's in the dictionary
        if len(keyindex) > 0:
            # put a large number in there
            eta[topic,keyindex[0]] = 1e7

    # normalize so that the probabilities sum to 1 over all topics
    eta = np.divide(eta, eta.sum(axis=0))

    return eta


corp = [preprocess(line) for line in txt]
dictionary = gensim.corpora.Dictionary(corp)

apriori_original = {'banana': 0, 'broccoli': 0, 'munch': 0,
                    'cute': 1, 'kitten': 1}
eta = create_eta(apriori_original, dictionary, 2)
test_eta(eta, dictionary, 2)
