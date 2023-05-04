import json
import textattack

data_path = "../dataset/test.json"


with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    examples = []
    for example in data:
        text = example["text"]
        label = example["fake"]
        tokens = text.split()
        if len(tokens) > 100: continue
        # append a tuplemy
        if label==1: examples.append((text, label))

dataset = textattack.datasets.Dataset(examples)