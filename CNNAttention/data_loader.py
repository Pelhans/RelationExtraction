#!/usr/bin/env python
# coding=utf-8
import json
import numpy as np
from collections import OrderedDict
import random

class BatchGenerator(object):
    """Construct a Data generator. THe input is json files
        file_name: Json file storing the data in the following format
            [
                {
                    'sentence': 'Bill Gates is the founder of Microsoft .',
                    'head': {'word': 'Bill Gates', ...(other information)},
                    'tail': {'word': 'Microsoft', ...(other information)},
                    'relation': 'founder'
                },
                ...
            ]
            #*IMPORTANT: In the sentence part, words and punctuations should be separated by blank spaces.
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        rel2id_file_name: Json file storing relation-to-id diction in the following format
            {
                'NA': 0
                'founder': 1
                ...
            }
            #*IMPORTANT: make sure the id of NA is 0.
    """

    def __init__(self, file_name, word_vec_file_name, rel2id_file_name, mode, max_length=120, batch_size=1, shuffle=True):
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ori_data = json.load(open(file_name, "r"))
        self.rel2id = json.load(open(rel2id_file_name))
        word_vec = json.load(open(word_vec_file_name))
        self.idx = 0
        self.instance_vec = []
        self.pos1 = []
        self.pos2 = []
        self.bag_data = []
        
        # Build word dicttionary
        self.word_dict = {w["word"] : w["vec"] for w in word_vec}
        self.word_dict["UNK"] = len(word_vec)
        self.word_dict["BLANK"] = len(word_vec) + 1
        self.ori_data.sort(key=lambda x: x["head"]["word"] + "#" + x["tail"]["word"] + "#" + x["relation"])

        # Convert each instance into vector
        for idx, instance in enumerate(self.ori_data):
            words = instance["sentence"].strip().split()
            vecs = [self.word_dict[w] if w in self.word_dict else self.word_dict["UNK"]  for w in words ]
            vecs = self._padding(vecs)
            self.instance_vec.append(vecs)

        # Calculate the distance of a word from an entity
        self.pos1 = [self._cal_distance(instance, pos="head") for instance in self.ori_data]
        self.pos2 = [self._cal_distance(instance, pos="tail") for instance in self.ori_data]

        # Package an instance of the same relationship
        self.bag_data, self.label = self._scope(self.ori_data)
        self.out_order = self.bag_data.keys()
        if self.shuffle:
            random.shuffle(self.out_order)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch(self.batch_size)

    def next_batch(self, batch_size):
        if self.idx >= len(self.bag_data):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.out_order)
            raise StopIteration
        
        batch_data = {}
        idx0 = self.idx
        idx1 = self.idx + batch_size
        if idx1 > len(self.out_order):
            idx1 = len(self.out_order)
        self.idx = idx1
        bag = []
        label = []
        for i in range(idx0, idx1):
            bag.append(self.bag_data[self.out_order[i]])
            label.append(self.label[self.out_order[i]])
        # If not enough for batch_size,fill it with 0
        for i in range(batch_size - (idx1-idx0)):
            bag.append(np.zeros((1, self.max_length), dtype=np.int32))
            label.append(0)
#        batch_data["instances"] = np.concatenate(bag)
        batch_data["instances"] = bag
        batch_data["labels"] = label
        return batch_data

    def _scope(self, instance):
#        bag_rel_vec = OrderedDict()
        bag_rel_vec = {}
        label = []
        start = 0
        end = 0
        last_key = ""
        for idx, x in enumerate(instance):
            key = x["head"]["word"] + "#" + x["tail"]["word"] + "#" + x["relation"]
            if last_key != key:
                relid = self.rel2id[x["relation"]] if x["relation"] in self.rel2id else self.rel2id["NA"]
#                bag_rel_vec[relid] = [self.instance_vec[i] for i in range(start, end)]
                bag_rel_vec[len(label)] = [self.instance_vec[i] for i in range(start, end)]
                label.append(relid)
                start = idx
                last_key = key
            end = idx
        return bag_rel_vec, label

    def _cal_distance(self, instance, pos="head"):
        # Avoid situations where the target entity is part of another word
        sentence = instance["sentence"]
        words = sentence.strip().split()
        # char_pos is the position of the character level
        char_pos = sentence.find(" " + instance[pos]["word"] + " ")
        if char_pos != -1:
            char_pos += 1
        else:
            if sentence[:len(instance[pos]["word"])+1] == instance[pos]["word"] + " ":
                char_pos = 0
            elif sentence[-len(instance[pos]["word"])-1:] == " " + instance[pos]["word"]:
                char_pos = len(sentence) - len(instance[pos]["word"])
            else:
                char_pos = 0
        tmp_p = 0
        for idx,w in enumerate(words):
            if tmp_p == char_pos:
                pos = idx
                char_pos = -1
            tmp_p += len(w) + 1
        pos = max_length - 1 if pos > self.max_length else pos
        words = self._padding(words, content="BLANK")
        distance_vec = [pos-i+self.max_length for i in range(len(words))]
        return distance_vec

    def _padding(self, vec, content=0):
        """ padding the vector with content, vector's shape should be (1,).
        if content = self.word_dict["BLANK"], we will padding it with BLANK id
        else content = 0, we will padding with 0
        """
        if len(vec) > self.max_length:
            vec = vec[:self.max_length]
        else:
            vec.extend(np.full(self.max_length-len(vec), content))
        return vec


if __name__ == "__main__":
    batch = BatchGenerator("../data/mini/train.json", "../data/mini/word_vec.json", "../data/mini/rel2id.json", 0)
    print batch.next_batch(2)
