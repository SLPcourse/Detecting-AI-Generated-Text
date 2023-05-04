import json
import pandas as pd
from tqdm import tqdm

def aug_process(augmenter, data_path, mode, aug_type, flag = 1, count = 5000, limit = 100):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(f"output/{mode}_{aug_type}_{count}_{flag}.json", "w", encoding="utf-8") as f:
        print("Create file successfully!")

    examples = []
    count0, count1 = 0, 0
    # flag = 1
    for example in data:
        question = example["question"]
        text = example["text"]
        label = example["fake"]

        tokens = text.split()
        if len(tokens) > limit: continue

        if label and count1 < count: 
            examples.append([question, text, label])
            count1 += 1
        elif label == 0 and count0 < count and flag == 0:
            examples.append([question, text, label])
            count0 += 1
        else:
            continue

    # print(len(examples))
    aug_examples = []

    for example in tqdm(examples, desc=f"{mode}_{aug_type}_{count}"):
            question = example[0]
            text = example[1]
            label = example[2]
            # print(text)
            if label:
                candidates = augmenter.augment(text)
                for s in candidates:
                    aug_examples.append({"question": question, "text": s, "fake": label})
  
        

    # Save samples to output file
    with open(f"output/{mode}_{aug_type}_{count}_{flag}.json", "w", encoding="utf-8") as f:
        for example in aug_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


# def aug_process(augmenter, data_path, mode, aug_type, count = 5000):
#     with open(data_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     aug_examples = []
#     count0, count1 = 0, 0
#     for idx in tqdm(len(data), desc=f"{mode}_{aug_type}_{count}"):
#         example = data[idx]
#         question = example["question"]
#         text = example["text"]
#         label = example["fake"]

#         # append a tuplemy
#         if label and count1 < count: 
#             candidates = augmenter.augment(text)
#             for s in candidates:
#                 aug_examples.append({"question": question, "text": s, "fake": label})
#         elif label == 0 and count0 < count:
#             candidates = augmenter.augment(text)
#             for s in candidates:
#                 aug_examples.append({"question": question, "text": s, "fake": label})
#         else:
#             continue

#     # Save samples to output file
#         with open(f"output/{mode}_{aug_type}_{count}.json", "w", encoding="utf-8") as f:
#             for example in aug_examples:
#                 f.write(json.dumps(example, ensure_ascii=False) + "\n")