#!/bin/bash

textattack attack --recipe textfooler --model-from-file custom/my_model.py --dataset-from-file custom/train_dataset_machine.py --num-examples 1000 | tee > outputs/textfooler_1000samples.txt
textattack attack --recipe textbugger --model-from-file custom/my_model.py --dataset-from-file custom/train_dataset_machine.py --num-examples 1000 | tee > outputs/textbugger_1000samples.txt
textattack attack --recipe bae --model-from-file custom/my_model.py --dataset-from-file custom/train_dataset_machine.py --num-examples 1000 | tee > outputs/bae_1000samples.txt
textattack attack --recipe deepwordbug --model-from-file custom/my_model.py --dataset-from-file custom/train_dataset_machine.py --num-examples 1000 | tee > outputs/deepwordbug_1000samples.txt
textattack attack --recipe pwws --model-from-file custom/my_model.py --dataset-from-file custom/train_dataset_machine.py --num-examples 1000 | tee > outputs/pwws_1000samples.txt