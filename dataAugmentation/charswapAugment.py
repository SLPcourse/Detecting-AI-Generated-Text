from textattack.augmentation import CharSwapAugmenter
import textattack
from augProcess import aug_process

train_data_path = "../dataset/train.json"
val_data_path = "../dataset/val.json"
test_data_path = "../dataset/test.json"


augmenter = CharSwapAugmenter(pct_words_to_swap=0.1, transformations_per_example=2)

aug_process(augmenter, train_data_path, "train", "charswap", flag=1, count=4000, limit=120)
aug_process(augmenter, val_data_path, "val", "charswap", flag=1, count=500, limit=120)
aug_process(augmenter, test_data_path, "test", "charswap", flag=1, count=1000, limit=120)



