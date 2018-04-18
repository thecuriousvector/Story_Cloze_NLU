from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import collections
import random
import math
from six.moves import xrange


''' Defining Parameters '''
hidden_units = 500	# m = 1000
embedding_size = 620 	# n = 620
vocab_size = 50004	#15356	# Unique 15355
batch_size = 80
layers = 1
epochs = 6		# 10 might be good
#max_src_len = 50	# ??

'''word_vec_size = 620
layer = 1
hidden dim = 1000
Optim = adadelta
learning_rate = 1.0
dropout = 0.0
brnn = True
batchsize = 80 '''

''' Reading Files '''
dev_ids_story = '../Data/Pad50/dev_pad_story'
dev_ids_ending = '../Data/Pad50/dev_pad_ending'
ids_story = '../Data/Pad50/test_pad_story'

''' Sentence counts for creating sentence matrix '''
total_sentences = 0                                                                 
max_src_len = 30
max_tar_len = 30 
half_src_len = (max_src_len/2) - 1
''' Counting total no of sentences '''   
with open (dev_ids_story, 'r') as sent_to_ids:
	sentences = sent_to_ids.readlines()
	total_sentences = len(sentences)
sent_to_ids.close()

sentence_matrix_src = np.full((total_sentences, max_src_len), 0, dtype=np.int32)
with open (dev_ids_story, 'r') as sent_to_ids:
	sentences = sent_to_ids.readlines()
	sentence_index = 0                 
	for sentence in sentences:      
		sentence_matrix_src[sentence_index, :max_src_len] = np.array(sentence.split(), dtype=np.int32)    
		sentence_index += 1
sent_to_ids.close()

sentence_matrix_tar = np.full((total_sentences, max_tar_len), 0, dtype=np.int32)
with open (dev_ids_ending, 'r') as sent_to_ids:
        sentences = sent_to_ids.readlines()
        sentence_index = 0
        for sentence in sentences:
                sentence_matrix_tar[sentence_index, :max_tar_len] = np.array(sentence.split(), dtype=np.int32)
                sentence_index += 1
sent_to_ids.close()

sentence_matrix_test = np.full((total_sentences, max_tar_len), 0, dtype=np.int32)
with open (test_ids_story, 'r') as sent_to_ids:
        sentences = sent_to_ids.readlines()
        sentence_index = 0
        for sentence in sentences:
                sentence_matrix_tar[sentence_index, :max_tar_len] = np.array(sentence.split(), dtype=np.int32)
                sentence_index += 1
sent_to_ids.close()

''' To generate batches of train and validation data '''
global last_visited_sent 
last_visited_sent = 0  # Stores the index of last sentence in the previous batch
def generate_batch(data_src, data_tar, batch_size):
	global last_visited_sent
	if (last_visited_sent+batch_size in range(len(data))):
		batch_src = data_tar[last_visited_sent:last_visited_sent+batch_size]
		batch_tar = data_src[last_visited_sent:last_visited_sent+batch_size]
	else:
		batch_src = data_src[last_visited_sent:]
		batch_tar = data_tar[last_visited_sent:]
	last_visited_sent  += batch_size
	return batch_src


graph = tf.Graph()
with graph.as_default():

	#x = tf.placeholder(tf.float32, shape=[None, vocab_size])
	#y = tf.placeholder(tf.float32, shape=[batch_size, 1])
	x = tf.placeholder(tf.int32, shape=[None, max_src_len])
	y = tf.placeholder(tf.float32, shape=[None, max_tar_len])
	sequence_length = tf.placeholder(tf.int32, [batch_size])
	iter_ind = tf.placeholder(tf.int32)
	isTest = tf.placeholder(tf.bool, shape=())
	''' embedding_matrix is shared between forward and backward directions '''
	#embedding_matrix = tf.Variable(tf.random_uniform([embedding_size, vocab_size], -1.0, 1.0))
	embedding_matrix = tf.get_variable('embedding_matrix', [batch_size, embedding_size])  # Check for transposed dimensions
	#embed_tensor = tf.nn.embedding_lookup(embedding_matrix, x)
	embed_tensor = tf.nn.embedding_lookup(embedding_matrix, x)
	
	embed_x_prod = tf.matmul(tf.transpose(embed_tensor[iter_ind]), tf.cast(x, tf.float32))

	''' Weights - Forward Direction '''
	W_f = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (hidden_units, embedding_size)), dtype=tf.float32)
	W_r_f = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (hidden_units, embedding_size)), dtype=tf.float32)
	W_z_f = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (hidden_units, embedding_size)), dtype=tf.float32)

	U_f = tf.get_variable("U_f", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))
	U_r_f = tf.get_variable("U_r_f", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))
	U_z_f = tf.get_variable("U_z_f", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))
	
	''' Calculating WEx_i - Forward Direction '''
	W_embed_x_prod_f = tf.matmul(W_f, embed_x_prod)

	''' Weights - Backward Direction '''
	W_b = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (hidden_units, embedding_size)), dtype=tf.float32)
	W_r_b = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (hidden_units, embedding_size)), dtype=tf.float32)
	W_z_b = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (hidden_units, embedding_size)), dtype=tf.float32)

	U_b = tf.get_variable("U_b", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))
	U_r_b = tf.get_variable("U_r_b", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))
	U_z_b = tf.get_variable("U_z_b", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))

	''' Calculating WEx_i - Backward Direction '''
	W_embed_x_prod_b = tf.matmul(W_b, embed_x_prod)
	
	
	#otho_matrix_intializer = tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001))
	# Implementing own GRU Cell as we need orthogonal intialization of matrices
	 #   for reset and update gates. Whereas, tf.nn.rnn.GRUCell() does random initialization 
	  #  of weight matrices 
	   # isForward is a boolean to indicate either a forward cell 
           # or a backward cell so that the corresponding weights can be chosen
            #If the state is at time i_1, None will be passed and will be initialized
            #with zeros''' 
		#'''def GRUCell_Modified(isForward, state):
		#if (state is None):
		#	state = tf.zeros([embedding_size], tf.float32)
		#if (isForward == True):
		#	r_i = tf.sigmoid(tf.add (tf.matmul(W_r_f, embed_x_prod), tf.matmul(U_r_f, state)))
		#	z_i = tf.sigmoid(tf.add (tf.matmul(W_z_f, embed_x_prod), tf.matmul(U_z_f, state)))
		#	current_state = tf.tanh(tf.add(W_embed_x_prod_f, tf.matmul(U_f, tf.multiply(r_i, state))))
		#	current_state =  """

	otho_matrix_intializer = tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001))
	forward_cell = tf.nn.rnn_cell.GRUCell(hidden_units, kernel_initializer=otho_matrix_intializer)
	init_state_f = forward_cell.zero_state([batch_size, hidden_units], tf.float32)		
	backward_cell = tf.nn.rnn_cell.GRUCell(hidden_units, kernel_initializer=otho_matrix_intializer)
	init_state_b = backward_cell.zero_state([batch_size, hidden_units], tf.float32)	
	bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, embed_tensor, 
					initial_state_fw=init_state_f, initial_state_bw= init_state_b, time_major=True)
	h_forward, h_backward = bi_outputs[0], bi_outputs[1]
	encoder_outputs = tf.concat([h_forward[0:half_src_len],h_backward[half_src_len:]], 1)

	 # Should we filter sentences of particular length?
	maximum_iterations = max_tar_len

	#(or)
	#maximum_iterations = tf.round(tf.reduce_max(max_src_len) * 2)

	attention = tf.contrib.seq2seq.BahdanauAttention( num_units=hidden_units, memory=encoder_outputs,normalize=False,name='BahdanauAttention' )
	cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units)
	final_cell = tf.contrib.seq2seq.AttentionWrapper( cell, attention, attention_layer_size=hidden_units / 2)

	Ws = tf.get_variable("Ws", [hidden_units,hidden_units], tf.float32, tf.orthogonal_initializer(np.random.normal(loc=0.0, scale=0.001)))
	s = tf.matmul(Ws,init_state_b)
	decoder_initial_state = tf.tanh(s)

	#y = tf.placeholder(tf.float32, shape=[None, max_target_len])
	decode_matrix = tf.get_variable('decode_matrix', [vocab_size, embedding_size])
	decoder_emb_inp = tf.nn.embedding_lookup(decode_matrix, y)
	helper = tf.contrib.seq2seq.TrainingHelper( decoder_emb_inp, max_target_len , time_major=True)
	def executeLoss():
		my_decoder = tf.contrib.seq2seq.BasicDecoder( final_cell, helper, decoder_initial_state)
		outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode( my_decoder, maximum_iterations=maximum_iterations)
                softmax_weights = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (batch_size, max_tar_len)), dtype=tf.float32)
                softmax_bias = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (batch_size, max_tar_len)), dtype=tf.float32)
                loss = tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, y, outputs, num_sampled, vocab_size)
        return loss
	def executeBeam():
		my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(final_cell, decoder_emb_inp, 1)
		outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode( my_decoder, maximum_iterations=maximum_iterations)	
	return outputs
	output_nmt = tf.cond(isTest,executeLoss,executeBeam)
	"""def executeLoss():
		softmax_weights = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (batch_size, max_tar_len)), dtype=tf.float32)
		softmax_bias = tf.Variable(np.random.normal(loc=0.0, scale=0.001, size = (batch_size, max_tar_len)), dtype=tf.float32)
		loss = tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, y, outputs, num_sampled, vocab_size)
	return loss
	def executeBeam():"""
		
	#tf.cond(isTest,executeLoss,executeBeam)
#def train_encoder():
#encoder = tf.global_variables_initializer()
with tf.Session(graph=graph) as session:
	nmt = tf.global_variables_initializer()
	for epoch in range(epochs):
		data_src, data_tar = generate_batch(sentence_matrix_src, sentence_matrix_tar, batch_size)
		loss_val = session.run(nmt, feed_dict = {x: data_src, y:data_tar,iter_ind:epoch, isTest:False})
		average_loss += loss_val
		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0
	
	translated_sentences = output_nmt.eval()

f = open('predictions', 'w')
for i in range(0,batch_size):
    format_str = ' '.join(['%d' % num for num in translated_sentences)
    f.write("%s ", format_str))
f.close()
	#print (session.run(encoder))
	#print (W.eval())

