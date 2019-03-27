#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import argparse
from network import Model
from data_loader import BatchGenerator
import time
import numpy as np
import sys, os
import sklearn.metrics

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='cnn_att')
parser.add_argument('--max_length', type=int, default=120)
parser.add_argument('--pos_embedding_dim', type=int, default=5)
parser.add_argument('--sentence_dim', type=int, default=230)
parser.add_argument('--lr', type=float, default=1.5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--save_epoch', type=int, default=10)
parser.add_argument('--test_epoch', type=int, default=10)
parser.add_argument('--train_file', type=str, default='/media/nlp/data/project/OpenNRE/data/nyt/train.json')
parser.add_argument('--test_file', type=str, default='/media/nlp/data/project/OpenNRE/data/nyt/test.json')
parser.add_argument('--word2id_file', type=str, default='/media/nlp/data/project/OpenNRE/data/nyt/word_vec.json')
parser.add_argument('--rel2id_file', type=str, default='/media/nlp/data/project/OpenNRE/data/nyt/rel2id.json')
#parser.add_argument('--train_file', type=str, default='/media/nlp/data/project/OpenNRE/data/mini/train.json')
#parser.add_argument('--test_file', type=str, default='/media/nlp/data/project/OpenNRE/data/mini/test.json')
#parser.add_argument('--word2id_file', type=str, default='/media/nlp/data/project/OpenNRE/data/mini/word_vec.json')
#parser.add_argument('--rel2id_file', type=str, default='/media/nlp/data/project/OpenNRE/data/mini/rel2id.json')
parser.add_argument('--summary_dir', type=str, default="./summary")
parser.add_argument('--ckpt_dir', type=str, default='./checkpoint')
args = parser.parse_args()


class Run(object):
    def __init__(self, batch_loader):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        optimizer = tf.train.GradientDescentOptimizer(args.lr)
        self.batch_loader = batch_loader
        self.sess = tf.Session(config=config)
        with tf.variable_scope(args.model):
            self.model = Model(self.batch_loader, args)
            self._loss, self.train_logit = self.model.cnn_att()
            grads = optimizer.compute_gradients(self._loss)
            self.train_op = optimizer.apply_gradients(grads)
            tf.add_to_collection("loss", self._loss)
            tf.add_to_collection("train_logit", self.train_logit)
            summary_writer = tf.summary.FileWriter(args.summary_dir, self.sess.graph)
            self.saver = tf.train.Saver(max_to_keep=None)
            self.sess.run(tf.global_variables_initializer())

    def train(self):
        test_loader = BatchGenerator(args.test_file, args.word2id_file, args.rel2id_file, mode="test", batch_size=args.batch_size)
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
                try:
                    batch_data = self.batch_loader.next_batch(args.batch_size)
                    iter_label = batch_data["rel"]
                except StopIteration:
                    break
                iter_loss, iter_logit, _train_op = self.model.run(batch_data, self.model, self.sess, run_list=[self._loss, self.train_logit, self.train_op], mode="train")
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

            if (epoch + 1)% args.save_epoch == 0:
                if not os.path.isdir(args.ckpt_dir):
                    os.mkdir(args.ckpt_dir)
                path = self.saver.save(self.sess, os.path.join(args.ckpt_dir, args.model))
                print("Finish storing in {}".format(path))

            if (epoch + 1)% args.test_epoch == 0:
                self.test(test_loader, self.model, self.sess)
            if (epoch + 1)% args.save_epoch == 0:
                if not os.path.isdir(args.ckpt_dir):
                    os.mkdir(args.ckpt_dir)
                path = self.saver.save(self.sess, os.path.join(args.ckpt_dir, args.model))
                print("Finish storing in {}".format(path))
        self.sess.close()

    def test(self, test_loader, model, sess, ckpt=None):
        tot_correct = 0
        tot_not_na_correct = 0
        tot = 0
        tot_not_na = 0
        entpair_tot = 0
        time_sum = 0
        test_result = []
        pred_result = []
        while True:
            time_start = time.time()
            feed_dict = {}
            try:
                test_data = test_loader.next_batch(args.batch_size)
                iter_label = test_data["rel"]
            except StopIteration:
                break
            iter_logit = model.run(test_data, model, sess, run_list=[self.train_logit], mode="test")
            t = time.time() - time_start
            time_sum += t
            iter_output = iter_logit.argmax(-1)
            iter_correct = (iter_output == test_data['rel']).sum()
            iter_not_na_correct = np.logical_and(iter_output == test_data['rel'],test_data['rel'] != 0).sum()
            tot_correct += iter_correct
            tot_not_na_correct += iter_not_na_correct
            tot += test_data['rel'].shape[0]
            tot_not_na += (test_data['rel'] != 0).sum()
            if tot_not_na > 0:
                sys.stdout.write("&&&&&&[TEST] not NA accuracy: %f, accuracy: %f\r"% (float(tot_not_na_correct) / tot_not_na, float(tot_correct) / tot))
                sys.stdout.flush()
            for idx in range(len(iter_logit)):
                for rel in range(1, test_loader.rel_tot):
                    test_result.append({'score': iter_logit[idx][rel], 'flag':test_data['multi_rel'][idx][rel]})
                    if test_data['entpair'][idx] != "None#None":
                        pred_result.append({'score': float(iter_logit[idx][rel]),'entpair': test_data['entpair'][idx].encode('utf-8'), 'relation': rel})
                entpair_tot += 1
        sorted_test_result = sorted(test_result, key=lambda x: x['score'])
        prec = []
        recall = []
        correct = 0
        for i, item in enumerate(sorted_test_result[::-1]):
            correct += item['flag']
            prec.append(float(correct) / (i + 1))
            recall.append(float(correct) / test_loader.relfact_tot)
        auc = sklearn.metrics.auc(x=recall, y=prec)
        print("\n[TEST] auc: {}".format(auc))
        print("Finish testing")
                

if __name__ == "__main__":
    batch_loader = BatchGenerator(args.train_file, args.word2id_file, args.rel2id_file,mode="train", batch_size=args.batch_size)
    run = Run(batch_loader)
    run.train()
