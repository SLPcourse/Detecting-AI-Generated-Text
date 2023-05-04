#!/bin/bash

textattack attack --recipe textfooler --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/textfooler_200samples.txt
textattack attack --recipe textbugger --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/textbugger_200samples.txt
textattack attack --recipe bae --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/bae_200samples.txt
textattack attack --recipe deepwordbug --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/deepwordbug_200samples.txt
textattack attack --recipe pwws --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/pwws_200samples.txt

textattack attack --recipe textfooler --model-from-file custom/my_model.py --dataset-from-file custom/val_dataset_machine.py --num-examples 100 | tee > outputs/val/textfooler_100samples.txt
textattack attack --recipe textbugger --model-from-file custom/my_model.py --dataset-from-file custom/val_dataset_machine.py --num-examples 100 | tee > outputs/val/textbugger_100samples.txt
textattack attack --recipe bae --model-from-file custom/my_model.py --dataset-from-file custom/val_dataset_machine.py --num-examples 100 | tee > outputs/val/bae_100samples.txt
textattack attack --recipe deepwordbug --model-from-file custom/my_model.py --dataset-from-file custom/val_dataset_machine.py --num-examples 100 | tee > outputs/val/deepwordbug_100samples.txt
textattack attack --recipe pwws --model-from-file custom/my_model.py --dataset-from-file custom/val_dataset_machine.py --num-examples 100 | tee > outputs/val/pwws_100samples.txt