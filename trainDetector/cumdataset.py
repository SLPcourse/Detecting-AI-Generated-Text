import torch

# Definition of a custom dataset for the sequence classification task
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length=512):
        self.inputs = inputs # input ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Return the number of examples in the dataset
    def __len__(self):
        return len(self.inputs)

    # Return a single example and its corresponding label
    def __getitem__(self, index):
        input_ids = self.tokenizer.encode(
            self.inputs[index], add_special_tokens=True, max_length=self.max_length
        )
        label = self.labels[index]
        return input_ids, label