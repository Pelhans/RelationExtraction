#!/usr/bin/env python
# coding=utf-8
import json

def mini_data(json_file):
    ori_data = json.load(open(json_file, "r"))
    json.dump(ori_data[:20], open("mini_" + json_file, "w"))

if __name__ == "__main__":
    mini_data("word_vec.json")
