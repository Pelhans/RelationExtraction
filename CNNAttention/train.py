#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import argparse
from network import Model
from data_loader import BatchGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='cnn_att')
parser.add_argument('--max_length', type=int, default=120)
args = parser.parse_args()

def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    batch_gen = BatchGenerator("../data/mini/train.json", "../data/mini/word_vec.json", "../data/mini/rel2id.json", 0)
    with tf.Session(config=config) as sess:
        with tf.variable_scope("inputs"):
            x_inputs = tf.placeholder(tf.float32, shape=[None, None, args.max_length], name="x_input")
            y_inputs = tf.placeholder(tf.float32, shape=[None], name="y_inputs")

        with tf.variable_scope(args.model, reuse=None):
            model = Model()
            model.cnn_att(x_inputs, y_inputs, batch_gen.word_vec_mat)

if __name__ == "__main__":
    train()
