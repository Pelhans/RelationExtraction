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
    batch_loader = BatchGenerator("../data/mini/train.json", "../data/mini/word_vec.json", "../data/mini/rel2id.json", 0)
    with tf.Session(config=config) as sess:
        with tf.variable_scope("inputs"):
            word = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="word")
            pos1 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="pos1")
            pos2 = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="pos2")
            label = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="label")
            ins_label=tf.placeholder(dtype=tf.int32, shape=[None], name="ins_label")
            length = tf.placeholder(dtype=tf.int32, shape=[None], name="length")
            scope = tf.placeholder(dtype=tf.int32, shape=[batch_size, 2], name="scope")
            train_data_loader = batch_loader
            rel_tot = batch_loader.rel_tot
            word_vec_mat = batch_loader.word_vec_mat

        with tf.variable_scope(args.model, reuse=None):
            model = Model()
            model.cnn_att()

if __name__ == "__main__":
    train()
