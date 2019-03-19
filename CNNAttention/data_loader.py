#!/usr/bin/env python
# coding=utf-8
import json


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

    def __init__(self, file_name, word_vec_file_name, rel2id_file_name, mode, max_length=0):
        self.max_length = max_length
        self.ori_data = json.load(open(file_name, "r"))
        self.rel2id = json.load(open(rel2id_file_name))
        word_vec = json.load(open(word_vec_file_name))
        self.instance_vec = []
        self.pos1 = []
        self.pos2 = []
        
        # Build word dicttionary
        self.word_dict = {w["word"] : w["vec"] for w in word_vec}
        self.word_dict["UNK"] = len(word_vec)
        self.word_dict["BLANK"] = len(word_vec) + 1
        self.ori_data.sort(key=lambda x: x["head"]["word"] + "#" + x["tail"]["word"] + "#" + x["relation"])

        # Convert each instance into vector
        for idx, instance in enumerate(self.ori_data):
            words = instance["sentence"].strip().split()
            vecs = [self.word_dict[w] if w in self.word_dict else self.word_dict["UNK"]  for w in words ]
            self.instance_vec.append(vecs)

        # Calculate the distance of a word from an entity
        self.pos1 = [self._cal_distance(instance, pos="head") for instance in self.ori_data]
        self.pos2 = [self._cal_distance(instance, pos="tail") for instance in self.ori_data]

#        print "self.pos1: ", self.pos1

    def _cal_distance(self, instance, pos="head"):
        # Avoid situations where the target entity is part of another word
        sentence = instance["sentence"]
        print "sentence: ", len(sentence), sentence
        p1 = sentence.find(" " + instance[pos]["word"] + " ")
        if p1 != -1:
            p1 += 1
            print p1
        else:
            if sentence[:len(instance[pos]["word"])+1] == instance[pos]["word"] + " ":
                p1 = 0
            elif sentence[-len(instance[pos]["word"])-1:] == " " + instance[pos]["word"]:
                print "len(sentence) : ", len(sentence) 
                p1 = len(sentence) - len(instance[pos]["word"])
            else:
                p1 = 0
        distance_vec = [p1-i+self.max_length for i in range(len(sentence.strip().split()))]
        return distance_vec


if __name__ == "__main__":
    batch = BatchGenerator("../data/mini/train.json", "../data/mini/word_vec.json", "../data/mini/rel2id.json", 0)
