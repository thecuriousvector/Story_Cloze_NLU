from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import collections
import random
import math
from six.moves import xrange

input_file = '../Data/Mappings/embed_id_map_story'
output_file = '../Data/Mappings/embed_id_sorted_story'

count = 0
with open (output_file, 'w') as output:
	with open(input_file, 'r') as input_f:
		lines =  input_f.readlines()
		lines.sort(key = lambda line: int(line.split()[0]) if (len(line) > 1) else -100)
		print(len(lines))
		for index in range(len(lines)):
			#if (len(lines[index]) > 0):
			output.write(lines[index])
			output.write('\n')
input_f.close()
output.close()
