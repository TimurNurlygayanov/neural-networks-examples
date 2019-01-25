#!/usr/bin/python3
# -*- encoding=utf8 -*-

"""
 This is pretty naive and simple spell checker for Russian texts.

 The code was taken from this article:
    https://habr.com/ru/company/singularis/blog/358664/
    https://gist.github.com/Kwentar/a957d29f7370f896b691c82ff9ebe7d2

 Many thanks to the author of this code!
 I've modified it a little bit for my case:
   1) Education now based on Russian words dictionary instead of some
      Russian text
   2) Improved parameters of the model
   3) Removed unnecessary Keras dependency
"""

import os
import requests

import gensim
import pymorphy2
from tqdm import tqdm


MODEL_NAME = 'ru_spellcheck.model'
WORD_TAGS = {'NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN',
             'PRTF', 'PRTS', 'GRND'}

# Wrong Russian words to check the model:
WRONG_WORDS = ['челавек', 'стулент', 'студечнеский', 'чиловенчость',
               'учавствовать', 'тактка', 'вообщем', 'симпотичный', 'зделать',
               'сматреть', 'алгаритм', 'ложить']
# Correct words:
CORRECT_WORDS = ['человек', 'студент', 'студенческий', 'человечность',
                 'участвовать', 'тактика', 'вообще', 'симпатичный', 'сделать',
                 'смотреть', 'алгоритм', 'положить']

if not os.path.isfile(MODEL_NAME):
    # Download Russian dictionary:
    result = requests.get('https://raw.githubusercontent.com/danakt/'
                          'russian-words/master/russian.txt')
    result.encoding = 'windows-1251'
    words_for_eduction = result.text

    # Prepare dictionary words for eduction:
    # Note: here we use only first 10000 words, but
    # for real cases we need to use the whole list of words!
    sentences = words_for_eduction.split('\n')[:10000] + CORRECT_WORDS

    morph = pymorphy2.MorphAnalyzer()

    for i in tqdm(range(len(sentences))):

        word = sentences[i]
        sentence = []

        p = morph.parse(word)[0]
        if p.tag.POS in WORD_TAGS:
            sentence.append(p.normal_form)

        sentences[i] = sentence
    sentences = [x for x in sentences if x]

    # Training the model (it can take 2+ hours!):
    model = gensim.models.FastText(sentences, size=300,
                                   window=1, min_count=1, sg=1, iter=35,
                                   min_n=1, max_n=6)
    model.init_sims(replace=True)

    # Save model to the disk (it will take 200+ Mb on disk!)
    model.save(MODEL_NAME)

    print('Neuron network model is ready!')

else:

    # Load model from file:
    model = gensim.models.FastText.load(MODEL_NAME)
    model.init_sims(replace=True)


spell_checker = model.wv

# Check what options model will suggest to correct each wrong word:
for index, word in enumerate(WRONG_WORDS):
    if word in spell_checker:
        print('\n ---- \n\n {0}'.format(word))

        for i in spell_checker.most_similar(positive=[word], topn=20)[:5]:
            print(i[0], i[1])
