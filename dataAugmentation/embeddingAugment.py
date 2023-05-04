from textattack.augmentation import EmbeddingAugmenter
import textattack
from augProcess import aug_process

train_data_path = "../dataset/train.json"
val_data_path = "../dataset/val.json"
test_data_path = "../dataset/test.json"

augmenter = EmbeddingAugmenter(pct_words_to_swap=0.1, transformations_per_example=2)
augmenter.fast_augment = True

aug_process(augmenter, train_data_path, "train", "embAug", flag=1, count=4000)
aug_process(augmenter, val_data_path, "val", "embAug", flag=1, count=500)
aug_process(augmenter, test_data_path, "test", "embAug", flag=1, count=1000)



