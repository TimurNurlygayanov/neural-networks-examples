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


# TODO:
# Create our own dictionary based on the text messages in chat.
# Not only replace based on neural network, but also based on
# simple dictionary.

class Checker(object):

    model = None
    words_dict = []
    learned_words_dict = []
    model_name = 'ru_spellcheck.model'
    tags = {'NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN',
            'PRTF', 'PRTS', 'GRND'}

    def check_word_in_dict(self, word, ignore='[A-Za-z0-9]'):
        """ Check only one word in the dict. """

        if re.search(ignore, word):
            return True

        if not self.words_dict:
            # Download Russian dictionary:
            russian_words_link = ('https://raw.githubusercontent.com/danakt/'
                                  'russian-words/master/russian.txt')
            res = requests.get(russian_words_link)
            res.encoding = 'windows-1251'
            self.words_dict = res.text.split('\n')

            self.words_dict = [str(w).strip() for w in self.words_dict]

        return word in self.words_dict

    def text_to_word_sequence(self, text):
        """ Makes the array of words from the text. """

        ignore_chars = r"[!@#$%^&*(){};:,./<>?\|`~=-_+]"

        text = re.sub(ignore_chars, ' ', text)

        result = [str(word).strip() for word in text.split(' ')
                  if word and word not in ignore_chars]

        return result

    def learn(self, text):
        """ Learn sentences patterns based on some text. """

        text = str(text).lower()
        sentences = []

        # Normalization
        for line in text.split('\n'):
            sentences.append(self.text_to_word_sequence(line))

        morph = pymorphy2.MorphAnalyzer()

        for i in tqdm(range(len(sentences))):
            sentence = []

            for word in sentences[i]:
                p = morph.parse(word)[0]
                if p.tag.POS in self.tags:
                    sentence.append(p.normal_form)

                if text.count(word) > 5:
                    word = word.strip()
                    if word not in self.learned_words_dict:
                        self.learned_words_dict.append(word)

            sentences[i] = sentence
        sentences = [x for x in sentences if x]

        # Training the model (it can take 2+ hours):
        model = gensim.models.FastText(sentences, size=300,
                                       window=4, min_count=3, sg=1,
                                       iter=100, min_n=3, max_n=6)
        model.init_sims(replace=True)

        # Save model to the disk (it will take 200+ Mb on disk)
        model.save(self.model_name)

        self.words_dict += self.learned_words_dict

    def spellcheck(self, text):
        """ Correct the text. """

        words = self.text_to_word_sequence(text)

        if not self.model:
            try:
                # Load model from file:
                model = gensim.models.FastText.load(self.model_name)
                model.init_sims(replace=True)
                self.model = model.wv
            except:
                print('Call "learn" first to teach the spellchecker!')

        for word in words:
            if not self.check_word_in_dict(word):

                try:
                    correct_words = self.model.most_similar(positive=[word])

                    if correct_words[0][1] > 0.8:
                        text = text.replace(' {0} '.format(word),
                                            ' {0} '.format(str(correct_words[0][0])),
                                            1)
                except:
                    pass  # Ignore any words if model can't find any
                          # options to replace it.

        return text
