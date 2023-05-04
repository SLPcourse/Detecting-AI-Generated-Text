### merge adversarial data and original data together, and shuffle
import json
import random

def extract_data(data_path, mode, count, limit = 100, mul = 1):
    with open(data_path + f"/{mode}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        # json.load(file) or [json.loads(line) for line in file], line is a str

    examples = []
    count0 = 0
    for example in data:
        question = example["question"]
        text = example["text"]
        label = example["fake"]

        tokens = text.split()
        if len(tokens) > limit: continue

        if label == 0 and count0 < count*mul:
            examples.append({"text": text, "fake": label})
            count0 += 1
        else:
            continue
    return examples

data_path = "../dataset"
for recipe in ['textfooler', 'textbugger', 'bae', 'deepwordbug', 'pwws']:
    ## train
    with open(f"outputs/train/{recipe}_1000samples_fake.txt", "r", encoding="utf-8") as f:
        fake_data = [json.loads(line) for line in f]
    human_data = extract_data(data_path, "train", count=1000, limit=100, mul=1)
    data = fake_data + human_data
    random.shuffle(data)

    print(f"train {recipe} length = {len(data)}")
    with open(data_path + f"/train_{recipe}_1000.json", "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


    ## val
    with open(f"outputs/val/{recipe}_100samples_fake.txt", "r", encoding="utf-8") as f:
        fake_data = [json.loads(line) for line in f]
    human_data = extract_data(data_path, "val", count=100, limit=100, mul=1)
    data = fake_data + human_data
    random.shuffle(data)

    print(f"val {recipe} length = {len(data)}")
    with open(data_path + f"/val_{recipe}_100.json", "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    
    ## test
    with open(f"outputs/test/{recipe}_200samples_fake.txt", "r", encoding="utf-8") as f:
        fake_data = [json.loads(line) for line in f]
    human_data = extract_data(data_path, "test", count=200, limit=100, mul=1)
    data = fake_data + human_data
    random.shuffle(data)

    print(f"test {recipe} length = {len(data)}")
    with open(data_path + f"/test_{recipe}_200.json", "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")