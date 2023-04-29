from textattack.augmentation import WordNetAugmenter
import textattack
from augProcess import aug_process

train_data_path = "../dataset/train.json"
val_data_path = "../dataset/val.json"
test_data_path = "../dataset/test.json"

augmenter = WordNetAugmenter(transformations_per_example=2)

aug_process(augmenter, train_data_path, "train", "wordnet", 5000)
aug_process(augmenter, val_data_path, "val", "wordnet", 500)
aug_process(augmenter, test_data_path, "test", "wordnet", 1000)



