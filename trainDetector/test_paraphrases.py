from cumdataset import CustomDataset
import json
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup


def load_merge_data(input_file, output_file, num):
    df = pd.read_csv(input_file, encoding="utf-8")
    
    samples = []
    count = 0
    for index, data in df.iterrows():
        if count >= num: break
        sample_0 = {"text": data["text"], "fake": 0}
        sample_1 = {"text": data["paraphrases"][0], "fake": 1}
        samples.append(sample_0).append(sample_1)
        count += 2
    
    # Save samples to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

class ModelTester():
    def __init__(
        self,
        test_file,
        model_name="bert-base-uncased",
        test_model_path = "best_model.pt"
    ):
        # Load the test data from JSON files
        self.test_data = self.load_data(test_file)
        # Instantiate a tokenizer and a pre-trained model for sequence classification
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name).cuda()
        self.test_model_path = test_model_path

    # Load data from a JSON file and return a list of examples
    def load_data(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        examples = []
        for example in data:
            text = example["text"]
            label = example["fake"]
            examples.append({"text": text, "label": label})
        return examples

    # Tokenize the inputs and labels and return them as two lists
    def tokenize_inputs(self, data):
        inputs = []
        labels = []
        for example in data:
            input_ids = self.tokenizer.encode(
                example["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=512,
            )
            inputs.append(input_ids)
            labels.append(example["label"])
        return inputs, labels

    # Function to pad input sequences and return them in a batch
    def collate_fn(self, batch):
        input_ids = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        # Get the maximum length of the input sequences in the batch
        max_length = max(len(ids) for ids in input_ids)
        input_ids_padded = []
        attention_masks = []
        # Pad the input sequences and create attention masks
        for ids in input_ids:
            padding = [0] * (max_length - len(ids))
            input_ids_padded.append(ids + padding)
            attention_masks.append([1] * len(ids) + padding)
        # Return the inputs and labels as a dictionary and a tensor, respectively
        inputs = {
            "input_ids": torch.tensor(input_ids_padded),
            "attention_mask": torch.tensor(attention_masks),
        }
        return inputs, torch.tensor(labels)

    # Evaluate the model on a given dataloader
    def evaluate_model(self, test_loader):
        # Set the model to evaluation mode and initialize the true and predicted labels
        self.model.eval()
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            # Iterate over the batches in the dataloader and get the model outputs
            for inputs, labels in test_loader:
                outputs = self.model(
                    inputs["input_ids"].cuda(),
                    attention_mask=inputs["attention_mask"].cuda(),
                )
                # Append the true and predicted labels to their respective lists
                true_labels.extend(labels)
                predicted_labels.extend(torch.argmax(outputs.logits, dim=1).cpu())
        # Calculate the classification report and the accuracy of the model
        report = classification_report(true_labels, predicted_labels, digits=4)
        return (
            report,
            torch.sum(torch.tensor(true_labels) == torch.tensor(predicted_labels))
            / len(true_labels),
        )

    def do_evaluate(self):
        # Tokenize the inputs and labels for the test dataset
        test_inputs, test_labels = self.tokenize_inputs(self.test_data)

        # Create dataloaders for the test dataset
        test_dataset = CustomDataset(test_inputs, test_labels, self.tokenizer)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )

        # Load the fine-tuned model 
        self.model.load_state_dict(torch.load(self.test_model_path))
        test_report, test_accuracy = self.evaluate_model(test_loader)
        # Print the best accuracy and the classification report for the test dataset
        print(f"Best accuracy: {test_accuracy:.4f}")
        print(test_report)

if __name__ == "__main__":
    load_merge_data("dataset/chatgpt_paraphrases.csv", "dataset/chatgpt_paraphrases.json")
    tester = ModelTester(
        "dataset/chatgpt_paraphrase.json"
    )
    tester.do_evaluate()