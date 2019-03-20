#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import argparse
from network import Model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='cnn_att')
parser.add_argument('--max_length', type=int, default=120)
args = parser.parse_args()

def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.variable_scope("inputs"):
            x_inputs = tf.placeholder(tf.float32, shape=[None, None, args.max_length], name="x_input")
            y_inputs = tf.placeholder(tf.float32, shape=[None], name="y_inputs")

        with tf.variable_scope(args.model, reuse=None):
            model = Model()

if __name__ == "__main__":
    train()
