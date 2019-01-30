#!/usr/bin/python3
# -*- encoding=utf8 -*-

#
# This code based on the information from this article:
# http://brandonrose.org/clustering
#
# Many thanks to the author of this great article!!!
#

import sys
sys.path.append('../')

import re
from telethon.sync import TelegramClient
from configparser import ConfigParser

from termcolor import colored

import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.manifold import MDS


from spell_checker_library.spell_checker import Checker

my_checker = Checker()


# Read Telegram private key from the local file:
config = ConfigParser()
config.read('/Users/timurnurlygayanov/.config.ini')


def get_conf_param(parameter, default_value):
    """ This function reads and returns the value of parameter
        from configuration file.
    """

    result = config.get('DEFAULT', parameter)
    return result or default_value


# Read all parameters from config file:
name = get_conf_param('name', '')
api_id = get_conf_param('api_id', '')
api_hash = get_conf_param('api_hash', '')
chat = get_conf_param('chat', '')

ALL_MESSAGES = []
ALL_QUESTIONS = []

with TelegramClient(name, api_id, api_hash) as client:
    for message in client.iter_messages(chat, limit=4000):
        if message.text:
            ALL_MESSAGES.append(str(message.text))

for q in ALL_MESSAGES:
    msg = q.replace('.', '\n').replace('(', '\n').replace(')', '\n')
    msgs = msg.replace('"', '\n').lower().split('\n')

    for m in msgs:
        if len(m) > 5:
            if '?' in str(m):
                for question in m.split('?')[:-1]:
                    ALL_QUESTIONS.append(question + '?')


print('Total messages: {0}'.format(len(ALL_MESSAGES)))
print('Questions found: {0}'.format(len(ALL_QUESTIONS)))


my_checker.learn('\n'.join(ALL_MESSAGES))


for i, q in enumerate(ALL_QUESTIONS):
    q = str(q).strip()
    q2 = my_checker.spellcheck(q)

    """
    if q2 != q:
        print(colored(q, 'red'))
        print(colored(q2, 'green'))

        print('\n --- \n\n')
    """

    ALL_QUESTIONS[i] = q2

# print(my_checker.learned_words_dict)
# exit(0)

print(len(ALL_QUESTIONS))
ALL_QUESTIONS = [q for q in ALL_QUESTIONS if ' ' in q]
print(len(ALL_QUESTIONS))
print('*'*20)

titles = ALL_QUESTIONS
synopses = ALL_QUESTIONS

stop_words = nltk.corpus.stopwords.words('russian')
stemmer = SnowballStemmer('russian')


def tokenize_only(text, correct_symbols='[А-Яа-яЁё]'):
    # First tokenize by sentence, then by word to ensure
    # that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    # Filter out any tokens not containing letters
    # (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search(correct_symbols, token):
            filtered_tokens.append(token)

    return filtered_tokens


def tokenize_and_stem(text):
    filtered_tokens = tokenize_only(text)

    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems


"""
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    # extend the 'totalvocab_stemmed' list
    # for each item in 'synopses', tokenize/stem
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized},
                           index = totalvocab_stemmed)
print('\n There are ' + str(vocab_frame.shape[0]) +
      ' items in vocab_frame')

# Define vectorizer parameters:
tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=20000,
                                   min_df=0.01, stop_words=stop_words,
                                   use_idf=True, tokenizer=tokenize_and_stem,
                                   ngram_range=(1, 6))

# Fit the vectorizer to synopses:
tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

terms = tfidf_vectorizer.get_feature_names()

dist = 1 - cosine_similarity(tfidf_matrix)
"""


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer=tokenize_and_stem,
                             stop_words=stop_words)

X = vectorizer.fit_transform(synopses)

num_clusters = 10

km = KMeans(n_clusters=num_clusters)
km.fit(X)

dist = 1 - cosine_similarity(X)


print('Similar questions:')
for i in ALL_QUESTIONS:
    predict_me = vectorizer.transform([i])
    if km.predict(predict_me) == 2:
        print(i + '\n=\n\n')


clusters = km.labels_.tolist()


MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e',
                  5: '#1b9077', 6: '#d95ff2', 7: '#0070b3', 8: '#572980', 9: '#9fa61e'}

#set up cluster names using a dict
cluster_names = {0: '1',
                 1: '2',
                 2: '3',
                 3: '4',
                 4: '5',
                 5: '6',
                 6: '7',
                 7: '8',
                 8: '9',
                 9: '10'}


# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

# group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(9, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    ax.tick_params( \
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelleft=False)

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
#for i in range(len(df)):
#    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

plt.show()  # show the plot
