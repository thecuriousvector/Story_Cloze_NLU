from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import collections
import random
import math
from six.moves import xrange


''' Declaring all the file paths and other variables '''
src_file = '../Data/Tokenized/dev_tok_story'
vocab_file = '../Data/Vocab/vocabulary_story'
src_map_file = '../Data/Mappings/dev_map_story'

#src_file = '../../Data/Tokenized/dev_tok.fr'
#vocab_file = '../../Data/Vocab/vocabulary.fr'
#src_map_file = '../../Data/Mappings/dev_map.fr'

word_to_id = collections.defaultdict(list)
vocabulary_set = set()

vocab_count = 1
with open (src_file, "r") as sentences:
        for sentence in sentences:
                for word in sentence.split():
                        word = word.strip()
                        #word word.lower() # added
                        if word not in vocabulary_set:
                                vocabulary_set.add(word)
                                word_to_id[word].append(vocab_count)
                                vocab_count += 1
sentences.close()

''' Adding UNK to vocabulary '''
UNK  = '<UNK>'
vocabulary_set.add(UNK)
word_to_id[UNK].append(vocab_count)
vocab_count += 1

with open (vocab_file, "w") as vocab_file:
        for word in word_to_id:
                vocab_file.write(word+" "+str(word_to_id[word][0])+"\n")
vocab_file.close()

with open (src_map_file, "w") as map_file:
        with open (src_file, "r") as sentences:
                for sentence in sentences:
                        for word in sentence.split():
                                word = word.strip()
                                #word = word.lower()
                                if (word in word_to_id):
                                        map_file.write(str(word_to_id[word][0])+" ")
                        map_file.write("\n")
        sentences.close()
map_file.close()


#vocab_file = '../Data/Vocab/vocabulary.en'
#src_map_file = '../Data/Mappings/dev_map.en'
#vocab_file = '../../Data/Vocab/vocabulary.fr'
#src_map_file = '../../Data/Mappings/dev_map.fr'
