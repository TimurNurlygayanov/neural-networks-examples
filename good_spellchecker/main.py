#!/usr/bin/python3
# -*- encoding=utf8 -*-

# This is just an example of usage of good Python
# Spellchecker. It is not fast, unfortunately, but
# the results are very good.

from autocorrect import Speller
from nltk.corpus import words


wrong_text = 'Look at this cute eylül munching on a las piece of broccoli.'

spell_checker = Speller(lang='en')

result = spell_checker(wrong_text)
print('autocorrect library: {0}'.format(result))


import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# lookup suggestions for single-word input strings
input_term = "las"  # misspelling of "members"
# max edit distance per lookup
# (max_edit_distance_lookup <= max_dictionary_edit_distance)
suggestions = sym_spell.lookup(input_term, Verbosity.CLOSEST,
                               max_edit_distance=2)
# display suggestion term, term frequency, and edit distance
for suggestion in suggestions:
    print(suggestion)

setofwords = set(words.words())

print('tapasvi' in setofwords)


import enchant

d = enchant.Dict('en_US')
print(d.check('anna'))
print(d.check('soc'))
