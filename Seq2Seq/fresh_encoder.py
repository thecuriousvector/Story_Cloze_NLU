from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#%matplotlib inline
import matplotlib.pyplot as plt
import time
import os
import urllib.request
#from tensorflow.models.rnn.ptb import reader
import ptb_iterator

src_file = '../Data/dev_tok_story'
dummy = 'dummy.txt'

file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name = 'tinyshakespeare.txt'
if not os.path.exists(file_name):
    urllib.request.urlretrieve(file_url, file_name)

with open(file_name,'r') as f:
	tok_data = f.read()
	print("Data length:", len(tok_data))
	
vocabulary = set(tok_data)
vocab_size = len(vocabulary)
idx_to_vocab = dict(enumerate(vocabulary))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
data = [vocab_to_idx[c] for c in tok_data]

'''for i in vocabulary:
	print (i)
for i in range(len(data)):
	print (data[i])
'''
data_array = np.array(data, dtype=np.int32)
data_len = len(data)
print (data_len)
'''for key, value in vocab_to_idx.items():
	print (key, value)'''
'''print (data[0])
def dummy():
	for i in range(1):
		yield ptb_iterator(data, 4, 1)

obj = dummy()
for i in  obj:
	print(obj[i])'''
