# show how to attack my custom model fine-tuned on HC3 dataset using TextAttack CLI
# with different recipes and 100 machine-generated examples and 100 human-written examples

# textfooler
textattack attack --recipe textfooler --model-from-file my_model.py --dataset-from-file custom/my_dataset_machine.py --num-examples 100
textattack attack --recipe textfooler --model-from-file my_model.py --dataset-from-file custom/my_dataset_human.py --num-examples 100

# textbugger
textattack attack --recipe textbugger --model-from-file my_model.py --dataset-from-file custom/my_dataset_machine.py --num-examples 100
textattack attack --recipe textbugger --model-from-file my_model.py --dataset-from-file custom/my_dataset_human.py --num-examples 100

# bert-based attack
textattack attack --recipe bae --model-from-file my_model.py --dataset-from-file custom/my_dataset_machine.py --num-examples 100
textattack attack --recipe bae --model-from-file my_model.py --dataset-from-file custom/my_dataset_human.py --num-examples 100

# checklist
textattack attack --recipe checklist --model-from-file my_model.py --dataset-from-file custom/my_dataset_machine.py --num-examples 100
textattack attack --recipe checklist --model-from-file my_model.py --dataset-from-file custom/my_dataset_human.py --num-examples 100

# deepwordbug
textattack attack --recipe deepwordbug --model-from-file my_model.py --dataset-from-file custom/my_dataset_machine.py --num-examples 100
textattack attack --recipe deepwordbug --model-from-file my_model.py --dataset-from-file custom/my_dataset_human.py --num-examples 100

# pruthi
textattack attack --recipe pruthi --model-from-file my_model.py --dataset-from-file custom/my_dataset_machine.py --num-examples 100
textattack attack --recipe pruthi --model-from-file my_model.py --dataset-from-file custom/my_dataset_human.py --num-examples 100

# pwws
textattack attack --recipe pwws --model-from-file my_model.py --dataset-from-file custom/my_dataset_machine.py --num-examples 100
textattack attack --recipe pwws --model-from-file my_model.py --dataset-from-file custom/my_dataset_human.py --num-examples 100

