from textattack.augmentation import CLAREAugmenter
import textattack
from augProcess import aug_process


class ClareAugmenterForDetector(CLAREAugmenter):

### to be written

train_data_path = "../dataset/train.json"
val_data_path = "../dataset/val.json"
test_data_path = "../dataset/test.json"


augmenter = ClareAugmenterForDetector(pct_words_to_swap=0.1, transformations_per_example=2)

aug_process(augmenter, train_data_path, "train", "clare", 5000)
aug_process(augmenter, val_data_path, "val", "clare",500)
aug_process(augmenter, test_data_path, "test", "clare", 1000)



