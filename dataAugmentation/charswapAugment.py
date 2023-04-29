from textattack.augmentation import CharSwapAugmenter
import textattack
from augProcess import aug_process

train_data_path = "../dataset/train.json"
val_data_path = "../dataset/val.json"
test_data_path = "../dataset/test.json"


augmenter = CharSwapAugmenter(transformations_per_example=2)

aug_process(augmenter, train_data_path, "train", "charswap", 5000)
aug_process(augmenter, val_data_path, "val", "charswap", 500)
aug_process(augmenter, test_data_path, "test", "charswap", 1000)



