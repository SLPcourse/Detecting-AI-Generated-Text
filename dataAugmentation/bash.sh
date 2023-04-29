#!/bin/bash
#openai_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

#python batchtest.py -key $openai_key --dataset HDFS

python embeddingAugment.py
python wordnetAugment.py
python charswapAugment.py
python edaAugment.py