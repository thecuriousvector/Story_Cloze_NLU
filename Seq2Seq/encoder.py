from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import collections
import random
import math
from six.moves import xrange
#sess = tf.InteractiveSession()

embedding_size = 200  	# m = 620
hidden_units = 5	# n = 1000 
vocab_size = 15355 	# 15355 6610

embed_id_file = '../Data/Mappings/embed_id_map_story'
dev_map_story = '../Data/Mappings/dev_map_story'

W = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (hidden_units, embedding_size)), dtype=tf.float32)
W_r = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (hidden_units, embedding_size)), dtype=tf.float32)
W_z = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (hidden_units, embedding_size)), dtype=tf.float32)

U = tf.get_variable("U", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))
U_r = tf.get_variable("U_r", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))
U_z = tf.get_variable("U_z", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))


''' Reading pre-trianed word embeddings '''
sentence_length = 0
max_length_sentence = 0
with open (dev_map_story, 'r') as sent_to_ids:
	sentences = sent_to_ids.readlines()
	sentence_length = len(sentences)
	for sentence in sentences:
		if (max_length_sentence < len(sentence)):
			max_length_sentence = len(sentence)
sent_to_ids.close()

print(sentence_length, max_length_sentence)	
sentence_matrix = np.full((sentence_length, max_length_sentence), 0, dtype=np.int32)
with open (dev_map_story, 'r') as sent_to_ids:
	sentences = sent_to_ids.readlines()
	sentence_index = 0
	for sentence in sentences:
		sent_array = np.array(sentence.split(), dtype=np.int32)
		sentence_matrix[sentence_index, :len(sentence)] = sent_array
		sentence_index += 1
sent_to_ids.close()

print (sentence_matrix[0])


encoder = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(encoder)
	print (session.run(encoder))
	print (W.eval())
	

''' Encoder Class '''
class Encoder(tf.nn.rnn_cell.RNNCell):
	num_units =  5        # n = 1000 
	
	with tf.variable_scope('GRU'):
            with tf.variable_scope("Gates"):
                ru = tf.nn.rnn_cell._linear([inputs, state], 2 * num_units, True, 1.0)
                ru = tf.nn.sigmoid(ru)
                r, u = tf.split(1, 2, ru)
	with tf.variable_scope("Candidate"):
                c = tf.nn.tanh(tf.nn.rnn_cell._linear([inputs, r * state], num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h
