import re
import json

def extractText(recipes, count, mode):
    
    for recipe in recipes:
        print(f"=== Processing {recipe} {mode} file ===")
        with open(f"outputs/{mode}/{recipe}_{count}samples.txt", "r") as f:
            lines = f.readlines()

        ### extract adversarial samples from attack outputs
        # 格式: result N + accuracy + 空行 + origin + 空行 + adversary + 空行*2 

        # search for the first line
        total_len = len(lines)
        start = 0
        step = 8
        for idx in range(total_len):
            if re.search(r'Result', lines[idx]): 
                start = idx
                break
        # start = 39
        # step = 8
        
        # total_len = 870
        origin_examples = []
        adv_examples = []

        while start < total_len:
            if total_len >= start + step:
                blocks = lines[start:(start+step)]
            else: blocks = lines[start:]

            # check if the last block valuable
            # if len(blocks) < 6 : continue

            # check if the format correct
            if re.search(r'Result', blocks[0]):
                # check if it is a successful attack
                if re.search(r"\[\[1 \([1-9]\d*%\)\]\] --> \[\[0 \([1-9]\d*%\)\]\]",blocks[1]):
                    origin_text = blocks[3].replace('[', '').replace(']', '')
                    adv_text = blocks[5].replace('[', '').replace(']', '')
                    origin_examples.append(origin_text)
                    adv_examples.append(adv_text)
                    # update start index
                start += step
            # if incorrect line
            else:
                # drop the last incorrect block
                if len(origin_examples): 
                    origin_examples.pop()
                    adv_examples.pop()
                for idx in range(start, total_len):
                    if re.search(r'Result', lines[idx]): 
                        start = idx
                        break
                    else:
                        if idx == total_len - 1: start = total_len

        examples = []
        for text in origin_examples:
            examples.append({'text': text, 'fake': 1})
        for text in adv_examples:
            examples.append({'text': text, 'fake': 1})


        with open(f"outputs/{mode}/{recipe}_{count}samples_fake.txt", "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        # write samples into files
        # with open(f"outputs/{mode}/{recipe}_origin_{count}samples.txt", "w") as f:
        #     for text in origin_examples:
        #         f.write(text)

        # with open(f"outputs/{mode}/{recipe}_adv_{count}samples.txt", "w") as f:
        #     for text in adv_examples:
        #         f.write(text)


recipes = ['textfooler', 'textbugger', 'bae', 'deepwordbug', 'pwws']
extractText(recipes, 1000, "train")
extractText(recipes, 100, "val")
extractText(recipes, 200, "test")
