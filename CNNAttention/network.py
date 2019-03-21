#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

class Model(object):
    def __init__(self):
        self.lr = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])
        
    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def cnn_att(self, x_inputs, y_inputs, word_vec_mat, max_length, keep_prob=1.0):
        max_length = 120
        pos_embedding_dim = 5
        kernel_size = 3
        stride_size = 1
        with tf.variable_scope("inputs"):
            self.x_inputs = tf.placeholder(tf.float32, shape=[None, None, max_length], name="x_input")
            self.y_inputs = tf.placeholder(tf.float32, shape=[None], name="y_inputs")
            self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, None, max_length], name="pos1")
            self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, None, max_length], name="pos2")
            self.word_vec_mat = word_vec_mat

        with tf.variable_scope("word_embedding"):
            word_embedding = tf.get_variable("word_embedding", initializer=word_vec_mat, dtype=tf.float32)
            x_embedding = tf.nn.embedding_lookup(word_embedding, self.x_inputs)
        
        with tf.variable_scope("pos_embedding"):
            pos1_embedding = tf.get_variable("pos1_embedding", [max_length, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            pos2_embedding = tf.get_variable("pos2_embedding", [max_length, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            p1_embedding = tf.nn.embedding_lookup(pos1_embedding, self.pos1)
            p2_embeddimg = tf.nn.embedding_lookup(pos2_embedding, self.pos2)
            pos_embeddimg = tf.concat([p1_embedding, p2_embeddimg], -1)

        input_embedding = tf.concat([x_embedding, pos_embeddimg], -1)

        with tf.variable_scope("cnn"):
            max_length = input_embedding.shape[1]
            x = self.__cnn_cell__(input_embedding, kernel_size, stride_size)
            x = __pooling__(x)
            x = actication(x)
            x = __dropout__(x, keep_prob)

        with tf.variable_scope("attention"):
            x = __dropout__(xm keep_prob=0.5)

        
    def __cnn_cell__(x, hidden_size=230, kernel_size=3, stride_size=1):
	x = tf.layers.conv1d(inputs=x, 
	                     filters=hidden_size, 
	                     kernel_size=kernel_size, 
	                     strides=stride_size, 
	                     padding='same', 
	                     kernel_initializer=tf.contrib.layers.xavier_initializer())
	return x

    def __dropout__(x, keep_prob=1.0):
        return tf.contrib.layers.dropout(x, keep_prob=keep_prob)
 
    def __pooling__(x):
        return tf.reduce_max(x, axis=-2)

