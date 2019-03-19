#!/usr/bin/env python
# coding=utf-8

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

    def __init__(self, file_name, word_vec_file_name, rel2id_file_name, mode):

