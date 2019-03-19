#!/bin/bash

wget https://cloud.tsinghua.edu.cn/f/11391e48b72749d8b60a/?dl=1 -O nyt.tar;

if [ $? -ne 0  ]; then
    echo "Download failed"
    exit 1
fi

tar -xvf nyt.tar;

if [ $? -ne 0  ]; then
    echo "Unzip failed"
    exit 1
fi
