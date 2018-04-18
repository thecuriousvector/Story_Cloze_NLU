from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import collections
import random
import math
from six.moves import xrange

''' Declaring all the file paths and other variables 
    Selects sentences from source file only with length <= specified length 
    Also, pads all sentences with length < specified length with zeros
    
    Since we need the sentences with corresponding line numbers in target language
    we create another file for target language too. After generation, we find out max length
    of target sentences and pad those with lesser lengths with zero'''

max_src_len = 30
max_tar_len = 0

src_map_file = '../Data/Mappings/dev_map_story'
src_len_file = '../Data/Length30/dev_map_30_story'
src_pad_file = '../Data/Pad30/dev_pad_story'

tar_map_file = '../Data/Mappings/dev_map_story'
tar_len_file = '../Data/Length30/dev_map_30_story'
tar_pad_file = '../Data/Pad30/dev_pad_story'

chosen_src_lines = []
line_count = 0
with open(src_map_file, 'r') as enFull:
	with open(src_len_file, 'w') as enLen:
		enFull_lines = enFull.readlines()
		for line in enFull_lines:
			if (len(line.split()) <= max_src_len):
				chosen_src_lines.append(line_count)
				enLen.write(line)
			line_count += 1
enLen.close()
enFull.close()

print ("Chosen Lines Count: "+str(len(chosen_src_lines)))

line_count = 0
with open(tar_map_file, 'r') as frFull:
	with open(tar_len_file, 'w') as frLen:
		frFull_lines = frFull.readlines()
		for src_idx in chosen_src_lines:
			line = frFull_lines[src_idx]
			frLen.write(line)
			if (len(line.split()) > max_tar_len):
				max_tar_len = len(line.split())
			line_count += 1
frLen.close()
frFull.close()

print ("Max Length of target sentence: ", max_tar_len)
print ("Target sentences count: ", line_count)

with open(src_len_file, 'r') as enLen:
	with open(src_pad_file, 'w') as enPad:
		enLen_lines = enLen.readlines()
		for line in enLen_lines:
			word_ids = line.split()
			if (len(word_ids) < max_src_len):
				word_ids.extend(['0'] * (max_src_len - len(word_ids)))
			#print(type(word_ids[0]))
			enPad.write(' '.join(word_ids))
			enPad.write("\n")
enLen.close()
enPad.close()

with open(tar_len_file, 'r') as frLen:
	with open(tar_pad_file, 'w') as frPad:
		frLen_lines = frLen.readlines()
		for line in frLen_lines:
			word_ids = line.split()
			if (len(word_ids) < max_tar_len):
				word_ids.extend(['0'] * (max_tar_len - len(word_ids)))
			frPad.write(' '.join(word_ids))
			frPad.write("\n")
frLen.close()
frPad.close()

# Refresh:
# rm src_len_file = '../Data/Length30/dev_map_30.en'
# rm src_pad_file = '../Data/Pad30/dev_pad.en'
# rm tar_len_file = '../Data/Length30/dev_map_30.fr'
# rm tar_pad_file = '../Data/Pad30/dev_pad.fr' 

