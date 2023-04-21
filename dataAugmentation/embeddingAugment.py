from textattack.augmentation import EmbeddingAugmenter


with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

augmenter = EmbeddingAugmenter()

machine_adv_examples = []
human_adv_examples = []
for example in data:
    question = example["question"]
    text = example["text"]
    label = example["fake"]

    candidates = augmenter.augment(text)
    # append a tuplemy
    if label==1: 
        for s in candidates:
            machine_examples.append({"question": question, "text": s, "fake": label})
    else: 
        for s in candidates:
            human_examples.append({"question": question, "text": s, "fake": label})

# Save samples to output file
    with open("machine_embAug.json", "w", encoding="utf-8") as f:
        for example in machine_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    with open("human_embAug.json", "w", encoding="utf-8") as f:
        for example in human_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
