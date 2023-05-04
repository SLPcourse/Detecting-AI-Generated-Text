#!/bin/bash

for model in textfooler textbugger bae deepwordbug pwws
do
for method in textfooler textbugger bae deepwordbug pwws
do
    textattack attack --recipe $method --model-from-file "custom/${model}_model.py" --dataset-from-file custom/test_dataset_machine.py --num-examples 100 | tee > "outputs/test/${model}_${method}_transfer.txt"
done
done