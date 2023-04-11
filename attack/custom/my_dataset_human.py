import json
import textattack

data_path = "/content/drive/MyDrive/Colab Notebooks/repos/Bert-based-Text-Detection/dataset/test.json"
# output_path = "my_dataset.py"


with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    examples = []
    for example in data:
        text = example["text"]
        label = example["fake"]
        # append a tuplemy
        if label==0: examples.append((text, label))

dataset = textattack.datasets.Dataset(examples)