#!/bin/bash
#openai_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

#python batchtest.py -key $openai_key --dataset HDFS

python charswapAugment.py
python embeddingAugment.py
python wordnetAugment.py
python edaAugment.py