#!/usr/bin/python3
# -*- encoding=utf8 -*-

# This is pretty naive and simple spell checker for Russian texts.
#
# The code was taken from this article:
#   https://habr.com/ru/company/singularis/blog/358664/
#   https://gist.github.com/Kwentar/a957d29f7370f896b691c82ff9ebe7d2
#
# Many thanks to the author of this code!
# I've modified it a little bit for my case:
#   1) Added check in Russian words dictionary and removed
#      attempts to replace the right words.
#   2) Removed unnecessary Keras dependency.
#   3) Created the library which can be used in other scripts.
#

import re
import requests

import gensim
import pymorphy2
from tqdm import tqdm


SPEELLCHECK_MODEL = None
SPEELLCHECK_WORDS_DICT = []
SPEELLCHECK_MODEL_NAME = 'ru_spellcheck.model'
IMPORTANT_TAGS = {'NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN',
                  'PRTF', 'PRTS', 'GRND'}


def check_word_in_dict(word):
    """ Check only one word in the dict. """

    global SPEELLCHECK_WORDS_DICT

    if not SPEELLCHECK_WORDS_DICT:
        # Download Russian dictionary:
        russian_words_link = ('https://raw.githubusercontent.com/danakt/'
                              'russian-words/master/russian.txt')
        res = requests.get(russian_words_link)
        res.encoding = 'windows-1251'
        SPEELLCHECK_WORDS_DICT = res.text.split('\n')

        SPEELLCHECK_WORDS_DICT = [str(w).strip() for w in SPEELLCHECK_WORDS_DICT]

    return word in SPEELLCHECK_WORDS_DICT


def text_to_word_sequence(text):
    """ Makes the array of words from the text. """

    text = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+ ]", '\n', text)

    result = [str(word).strip() for word in text.split('\n') if word]

    return result


def learn(text):
    """ Learn sentences patterns based on some text. """

    sentences = []

    # Normalization
    for line in text.split('\n'):
        sentences.append(text_to_word_sequence(line))

    morph = pymorphy2.MorphAnalyzer()

    for i in tqdm(range(len(sentences))):

        word = sentences[i]
        sentence = []

        p = morph.parse(word)[0]
        if p.tag.POS in IMPORTANT_TAGS:
            sentence.append(p.normal_form)

        sentences[i] = sentence
    sentences = [x for x in sentences if x]

    # Training the model (it can take 2+ hours):
    model = gensim.models.FastText(sentences, size=300,
                                   window=4, min_count=3, sg=1,
                                   iter=100, min_n=1, max_n=8)
    model.init_sims(replace=True)

    # Save model to the disk (it will take 200+ Mb on disk)
    print('Saving model to the disk...')
    model.save(SPEELLCHECK_MODEL_NAME)


def spellcheck(text):
    """ Correct the text. """

    global SPEELLCHECK_MODEL

    words = text_to_word_sequence(text)

    if not SPEELLCHECK_MODEL:
        try:
            # Load model from file:
            model = gensim.models.FastText.load(SPEELLCHECK_MODEL_NAME)
            model.init_sims(replace=True)
            SPEELLCHECK_MODEL = model.wv
        except:
            print('Call "learn" first to teach the spellchecker!')

    for word in words:
        if not check_word_in_dict(word):
            correct_words = SPEELLCHECK_MODEL.most_similar(positive=[word])
            text = text.replace(word, correct_words[0])

    return text
