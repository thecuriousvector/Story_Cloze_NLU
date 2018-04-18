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
src_file = '../Data/dev_tok_story'
embed_file = '../Data/dev_tok_embed.en.vec'
vocab_file = '../Data/vocabulary_story'
src_map_file = '../Data/Mappings/dev_map_story'
embed_id_file = '../Data/Mappings/embed_id_map_story'
word_to_id = collections.defaultdict(list)
vocabulary_set = set()
''' Generate a vocabulary file with word and id printed in each line '''
''' def generate_vocab_file():
	vocab_count = 0
	with open (src_file, "r") as sentences:
		for sentence in sentences:
			for word in sentence.split():
				word = word.strip()
				if word not in vocabulary_set:
					vocabulary_set.add(word)
					word_to_id[word].append(vocab_count)
					vocab_count += 1 '''

vocab_count = 0
with open (src_file, "r") as sentences:
	for sentence in sentences:
		for word in sentence.split():
			word = word.strip()
			#word word.lower() # added
			if word not in vocabulary_set:
				vocabulary_set.add(word)
				word_to_id[word].append(vocab_count)
				vocab_count += 1

'''for word in word_to_id:
	print (word, word_to_id[word][0])'''

'''with open (vocab_file, "w") as vocab_file:
	for word in word_to_id:
		vocab_file.write(word+" "+str(word_to_id[word][0])+"\n")
vocab_file.close()'''
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

isFirstLine = True
with open (embed_id_file, "w") as embed_id:
	with open (embed_file, "r") as embed:
		for line in embed:
			#print (line, line.split()[0])
			#if (isFirstLine == False):
			word = line.split()[0]
			#word = word.lower()
			if (word in word_to_id):
				id_line = line.replace(word, str(word_to_id[word][0]))
				embed_id.write(id_line)
				embed_id.write("\n")
			#else:
				#isFirstLine = False
embed.close()
embed_id.close()
'''
Total no of words in dev_tok.en = 137780 (wc -w ../Data/dev_tok.en)  
Unique words in dev_tok.en = 15355  (tr ' ' '\n' < ../Data/dev_tok.en | sort -u | wc -w)
'''
