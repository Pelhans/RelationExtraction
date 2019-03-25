#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import argparse
from network import Model
from data_loader import BatchGenerator
import time
import numpy as np
import sys, os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='cnn_att')
parser.add_argument('--max_length', type=int, default=120)
parser.add_argument('--pos_embedding_dim', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_epoch', type=int, default=60)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--summary_dir', type=str, default="./summary")
parser.add_argument('--checkpoint', type=str, default='./checkpoint')
args = parser.parse_args()

def train():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    optimizer = tf.train.GradientDescentOptimizer(args.lr)
    batch_loader = BatchGenerator("../data/mini/train.json", "../data/mini/word_vec.json", "../data/mini/rel2id.json", 0)
    with tf.Session(config=config) as sess:
        with tf.variable_scope(args.model):
            model = Model(batch_loader, args)
            _loss, train_logit = model.cnn_att()
            grads = optimizer.compute_gradients(_loss)
            train_op = optimizer.apply_gradients(grads)
            tf.add_to_collection("loss", _loss)
            tf.add_to_collection("train_logit", train_logit)
            summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)
            saver = tf.train.Saver(max_to_keep=None)
            sess.run(tf.global_variables_initializer())

        # Training
        best_metric = 0
        best_prec = None
        best_recall = None
        not_best_count = 0
        for epoch in range(args.max_epoch):
            print("#### Epoch {} ####".format(epoch))
            tot_correct = 0
            tot_not_na_correct = 0
            tot = 0
            tot_not_na = 0
            i = 0
            time_sum = 0
            while True:
                time_start = time.time()
                feed_dict = {}
                batch_data = batch_loader.next_batch(args.batch_size)
#                print "batch_data['word']: ", batch_data["word"]
                iter_label = batch_data["rel"]
                feed_dict.update({
                    model.word : batch_data["word"],
                    model.pos1: batch_data["pos1"],
                    model.pos2 : batch_data["pos2"],
                    model.label : batch_data["rel"],
                    model.ins_label: batch_data["ins_rel"],
                    model.scope : batch_data["scope"],
                    model.length : batch_data["length"]
                })
#                print "#" * 20, "type: ", type(_loss), type(train_logit), type(train_op)
                iter_loss, iter_logit, _train_op = sess.run([_loss, train_logit, train_op], feed_dict)
                t = time.time() - time_start
                time_sum += t
                iter_output = iter_logit.argmax(-1)
                iter_correct = (iter_output == iter_label).sum()
                iter_not_na_correct = np.logical_and(iter_output == iter_label, iter_label != 0).sum()
                tot_correct += iter_correct
                tot_not_na_correct += iter_not_na_correct
                tot += iter_label.shape[0]
                tot_not_na += (iter_label != 0).sum()
                if tot_not_na > 0:
                    sys.stdout.write("epoch %d step %d time %.2f | loss: %f, not NA accuracy: %f, accuracy: %f\r" % (epoch, i, t, iter_loss, float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot)) 
                    sys.stdout.flush()
                i += 1
            print("\nAverage iteration time: %f" % (time_sum / i))

            if (epoch + 1)% save_epoch == 0:
                if not os.path.isdir(ckpt_dir):
                    os.mkdir(ckpt_dir)
                path = saver.save(sess, os.path.join(args.ckpt_dir, args.model))
                print("Finish storing in {}".format(path))


if __name__ == "__main__":
    train()
