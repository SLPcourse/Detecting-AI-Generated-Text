#!/bin/bash

for method in textfooler textbugger bae deepwordbug pwws
do
    textattack attack --recipe $method --model-from-file "custom/${method}_model.py" --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/{$method}_attack_robust.txt
done

# textattack attack --recipe textfooler --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/textfooler_200samples.txt
# textattack attack --recipe textbugger --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/textbugger_200samples.txt
# textattack attack --recipe bae --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/bae_200samples.txt
# textattack attack --recipe deepwordbug --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/deepwordbug_200samples.txt
# textattack attack --recipe pwws --model-from-file custom/my_model.py --dataset-from-file custom/test_dataset_machine.py --num-examples 200 | tee > outputs/test/pwws_200samples.txt