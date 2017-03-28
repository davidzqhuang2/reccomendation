import tensorflow as tf
import numpy as np

class rec(object):

	def __init__(self, sentence_length,num_classes, vocab_size, embedding_size, filter_sizes,num_filters):

# Placeholders for input, output, and dropout

		self.input_x = tf.placeholder(tf.int32, [None, paragraph_length], name = "input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

		with tf.device('/cpu:0)', tf.name_scope("embedding"):
			W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name="W")
			self.embedded_chars = tf.nn.embedding_lookup(W,self.input_x)
			self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,-1)a

		pooled_outputs = []
		for i, filter_size in enumerate(filter_size):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
# Convolution Layer
			filter_shape = [filter_size,embedding_size, 1, num_filters]
			W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name = "W")	
