from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

import os

import re
import string
# Constants

TEXT_SIZE = 140
FEELINGS_SIZE = 10

vocabulary_size = 100000
embedding_size = 128

# Read Data

 

# Structure of Net

graph = tf.Graph()

with graph.as_default():
 inputs = tf.placeholder(tf.int32, shape=[134])
 a = tf.placeholder(tf.int32,shape=[])
 value_in = tf.placeholder(tf.float32, shape=[])
 dictionary = dict()
 rev_dictionary = dict()

 full = False
 with tf.device('/cpu:0'):
  state = tf.Variable(tf.random_normal([1,7,8], stddev=0.35))
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  inputs_r = tf.expand_dims(inputs,0,name=None)
  # Policy Net
  embed = tf.nn.embedding_lookup(embeddings, inputs_r)

  filter1 = tf.zeros([4,128,128])
  conv1 = tf.nn.conv1d(embed, filter1, 2, 'VALID')

  conv2 = tf.nn.conv1d(conv1, filter1, 2, 'VALID')

  dense = tf.layers.dense(conv2, 256, activation=tf.nn.relu, use_bias =True,kernel_initializer=None,bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)

  filter2 = tf.zeros([8,256,8])
  conv3 = tf.nn.conv1d(dense, filter2, 4, 'VALID')

  consolidate = tf.concat([conv3,state],1)

  dense2 = tf.layers.dense(consolidate, 256, activation=tf.nn.relu, use_bias =True,kernel_initializer=None,bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)
  print(dense2.get_shape())

  re_dense2 = tf.expand_dims(dense2,1,name=None,dim =None)

  # Get output
  filter3 = tf.zeros([1,4,128,256]) 
  tran1 = tf.nn.conv2d_transpose(value=re_dense2,filter=filter3,output_shape=[1,1,30,128],strides=[1,1,2,1],padding='VALID') 

  filter3 = tf.zeros([1,4,128,128]) 
  tran2 = tf.nn.conv2d_transpose(value=tran1,filter=filter3,output_shape=[1,1,62,128],strides=[1,1,2,1],padding='VALID')

  tran3 = tf.nn.conv2d_transpose(value=tran2,filter=filter3,output_shape=[1,1,126,128],strides=[1,1,2,1],padding='VALID')

  batch_array = tf.squeeze(tran3)

  normed_embedding = tf.nn.l2_normalize(embeddings, dim=1)
  normed_array = tf.nn.l2_normalize(batch_array, dim=1)
  cosine_similarity = tf.matmul(normed_array, tf.transpose(normed_embedding, [1, 0]))
  closest_words = tf.argmax(cosine_similarity, 1) # shape [batch_size], type int64

  output = tf.nn.embedding_lookup(embeddings,closest_words)


  # Get State
  state_pool = tf.nn.max_pool(value=re_dense2,ksize=[1,1,2,1],strides=[1,1,2,1],padding='VALID')

  dense_v1 = tf.layers.dense(inputs=state_pool,units=8, activation=None, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)
  
  state__1 = tf.squeeze(dense_v1)
  state__2 = tf.expand_dims(state__1,0,name=None,dim=None)
  state = state__2

  # Value Net
  def f1(): return tf.nn.embedding_lookup(embeddings,inputs)
  def f2(): return tf.nn.embedding_lookup(embeddings,closest_words) 
  embed_v_t = tf.cond(tf.less(0,a), f1,f2)
  embed_v = tf.expand_dims(embed_v_t,0,name=None,dim=None)

  filter1_v = tf.zeros([4,128,128])
  conv1_v = tf.nn.conv1d(embed_v, filter1_v, 2, 'VALID')

  conv2_v = tf.nn.conv1d(conv1_v, filter1_v, 2, 'VALID')

  dense_v = tf.layers.dense(conv2_v, 256, activation=tf.nn.relu, use_bias =True,kernel_initializer=None,bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)

  filter2_v = tf.zeros([8,256,8])
  conv3_v = tf.nn.conv1d(dense_v, filter2_v, 4, 'VALID')

  consolidate_v = tf.concat([conv3_v,state],1)

  dense2_v = tf.layers.dense(consolidate_v, 128, activation=tf.nn.relu, use_bias =True,kernel_initializer=None,bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)

  state_pool_v2_v = tf.expand_dims(dense2_v,1,name=None,dim =None)
  
  # Get Value
  re_conv1_v = tf.expand_dims(conv1_v,1,name=None,dim=None)
  re_conv2_v = tf.expand_dims(conv2_v,1,name=None,dim=None)
  re_conv3_v = tf.expand_dims(conv3_v,1,name=None,dim=None)
  dense_v1_v = tf.layers.dense(inputs=re_conv1_v,units=128, activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)
  dense_v2_v = tf.layers.dense(inputs=re_conv2_v,units=128, activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)
  dense_v3_v = tf.layers.dense(inputs=re_conv3_v,units=128, activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)
  dense_v4_v = tf.layers.dense(inputs=state_pool_v2_v,units=128, activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)

  consolidate2_v = tf.concat([dense_v1_v,dense_v2_v,dense_v3_v,dense_v4_v],2)

  dense_f_v = tf.layers.dense(inputs=consolidate2_v,units=8, activation=None, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, trainable=True, name=None, reuse=None)
   
  pool_f_v = tf.nn.max_pool(value=dense_f_v,ksize=[1,1,104,1],strides=[1,1,1,1],padding='VALID')
  value = tf.reduce_mean(pool_f_v)

  # Losses and Optimizers
  v_loss = tf.square(value-value_in)
  
  optimizer_value = tf.train.GradientDescentOptimizer(0.01).minimize(v_loss)

  p_loss = value
  
  optimizer_policy = tf.train.GradientDescentOptimizer(0.01).minimize(p_loss)
  
  init = tf.global_variables_initializer()

  saver = tf.train.Saver()

dictionary = dict()
rev_dictionary = dict()
dictionary[0]= "UNK"
rev_dictionary["UNK"] = 0
def live(): 
 with tf.Session(graph=graph) as session:
   init.run()
   saver.restore(session, "/tmp/chat.ckpt")
   print("Initialized")
   
   while True:
    cmd = raw_input("cmd: ")
    if cmd == "q":
     print("Exit")
     break
    if cmd == "w":
     save_path = saver.save(session, "/tmp/chat.ckpt")
     print("Model saved in file: %s" % save_path)
    if cmd == "wq":
     save_path = saver.save(session, "/tmp/chat.ckpt")
     print("Model saved in file: %s" % save_path)
     print("Exit")
     break
 
    inp = raw_input("You: ").split()
    v_input = tf.convert_to_tensor(inp)
    to_embed = list()
    for word in inp:
     print(word)
     if word in rev_dictionary:
      to_embed.append(rev_dictionary[word]) 
     elif len(dictionary)>=vocabulary_size:
      dictionary[len(dictionary)+1]= word
      rev_dictionary[word] = len(dictionary)
      to_embed.append(rev_dictionary[word]) 
     else:
      to_embed.append(0)
    to_embed = to_embed + [0 for _ in range(134-len(to_embed))]
    _, v_loss_val = session.run([optimizer_value, v_loss],feed_dict={inputs:to_embed, a:1, value_in:0})
    _, out, na = session.run([optimizer_policy, closest_words,batch_array],feed_dict={inputs:to_embed, a:-1,value_in:0})
    print(na)
    to =""
    for w in out:
     wor = dictionary[w]
     to += wor
     to += " "
    print(to)

def read():

 with tf.Session(graph=graph) as session:

  init.run()
  saver.restore(session, "/tmp/chat.ckpt")
  print("Initialized")

  with open("text8") as f:

   for line in iter(f):
    inp = line.split()
    to_embed = list()
    for word in iter(inp):
     to_embed = list()
     if word in rev_dictionary:
      to_embed.append(rev_dictionary[word]) 
     elif len(dictionary)>=vocabulary_size:
      dictionary[len(dictionary)+1]= word
      rev_dictionary[word] = len(dictionary)
      to_embed.append(rev_dictionary[word]) 
     else:
      to_embed.append(0)      
     to_embed = to_embed + [0 for _ in range(134-len(to_embed))]
     _, v_loss_val = session.run([optimizer_value, v_loss],feed_dict={inputs:to_embed, a:1, value_in:0})
     save_path = saver.save(session, "/tmp/chat.ckpt")

live()
