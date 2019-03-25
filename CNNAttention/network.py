#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, batch_loader, args):
        self.args = args
        self.batch_loader = batch_loader
        self.lr = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])
        
    def cnn_att(self, keep_prob=1.0):
        max_length = self.args.max_length
        batch_size = self.args.batch_size
        batch_loader = self.batch_loader
        pos_embedding_dim = self.args.pos_embedding_dim
        kernel_size = 3
        stride_size = 1
        with tf.variable_scope("inputs"):
            self.word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="word")
            self.pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="pos1")
            self.pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="pos2")
            self.label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="label")
            self.ins_label = tf.placeholder(dtype=tf.int32, shape=[None], name="ins_label")
            self.length = tf.placeholder(dtype=tf.int32, shape=[None], name="length")
            self.scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name="scope")
            self.train_data_loader = batch_loader
            self.rel_tot = batch_loader.rel_tot
            self.word_vec_mat = batch_loader.word_vec_mat

        with tf.variable_scope("word_embedding"):
            word_embedding = tf.get_variable("word_embedding", initializer=self.word_vec_mat, dtype=tf.float32)
            x_embedding = tf.nn.embedding_lookup(word_embedding, self.word)
        
        with tf.variable_scope("pos_embedding"):
            pos1_embedding = tf.get_variable("pos1_embedding", [max_length*2, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            pos2_embedding = tf.get_variable("pos2_embedding", [max_length*2, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            p1_embedding = tf.nn.embedding_lookup(pos1_embedding, self.pos1)
            p2_embeddimg = tf.nn.embedding_lookup(pos2_embedding, self.pos2)
            pos_embeddimg = tf.concat([p1_embedding, p2_embeddimg], -1)

        input_embedding = tf.concat([x_embedding, pos_embeddimg], -1)

        with tf.variable_scope("cnn"):
            max_length = input_embedding.shape[1]
            x = self.__cnn_cell__(input_embedding, kernel_size, stride_size)
            x = self.__pooling__(x)
            x = tf.nn.relu(x)
            x = self.__dropout__(x, keep_prob)

        with tf.variable_scope("attention"):
            x = self.__dropout__(x, keep_prob=0.5)
            bag_repre = []
            with tf.variable_scope("logit"):
                relation_matrix = tf.get_variable("relation_matrix", shape=[self.rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                bias = tf.get_variable("bias", shape=[self.rel_tot], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                current_relation = tf.nn.embedding_lookup(relation_matrix, self.ins_label)
                attention_logit = tf.reduce_sum(current_relation*x, -1)
            for i in range(self.scope.shape[0]):
                bag_hidden_mat = x[self.scope[i][0]:self.scope[i][1]]
                attention_score = tf.nn.softmax(attention_logit[self.scope[i][0]:self.scope[i][1]], -1)
                bag_repre.append(tf.squeeze(tf.matmul(tf.expand_dims(attention_score, 0), bag_hidden_mat)))
            bag_repre = tf.stack(bag_repre)
            bag_repre = self.__dropout__(bag_repre, keep_prob)
            _train_logit = self.__logit__(bag_repre, self.rel_tot, var_scope="att")

        with tf.variable_scope("loss"):
            weights = tf.nn.embedding_lookup(self._get_weights(), self.label)
            label_one_hot = tf.one_hot(indices=self.label, depth=self.rel_tot, dtype=tf.int32)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=label_one_hot, logits=_train_logit, weights=1.0)
            tf.summary.scalar("loss", loss)
        return loss, _train_logit


    def __logit__(self, x, rel_tot, var_scope=None):
        with tf.variable_scope(var_scope):
            relation_matrix = tf.get_variable('relation_matrix', shape=[rel_tot, x.shape[1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias', shape=[rel_tot], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            logit = tf.matmul(x, tf.transpose(relation_matrix)) + bias
        return logit
        
    def __cnn_cell__(self, x, hidden_size=230, kernel_size=3, stride_size=1):
	x = tf.layers.conv1d(inputs=x, 
	                     filters=hidden_size, 
	                     kernel_size=kernel_size, 
	                     strides=stride_size, 
	                     padding='same', 
	                     kernel_initializer=tf.contrib.layers.xavier_initializer())
	return x

    def __dropout__(self, x, keep_prob=1.0):
        return tf.contrib.layers.dropout(x, keep_prob=keep_prob)
 
    def __pooling__(self, x):
        return tf.reduce_max(x, axis=-2)

    def _get_weights(self):
        with tf.variable_scope("weights_table"):
            print("Calculating weights_table...")
            _weights_table = np.zeros((self.rel_tot), dtype=np.float32)
            for i in range(len(self.batch_loader.data_rel)):
                _weights_table[self.data_loader.data_rel[i]] += 1.0
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False, initializer=_weights_table)
            print("Finish calculating")
        return weights_table


