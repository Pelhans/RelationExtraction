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

    def cnn_att(self, x_inputs, y_inputs):
        pass

