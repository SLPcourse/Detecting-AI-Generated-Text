import json
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

# Definition of a custom dataset for the sequence classification task
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, attention_mask, labels):
        self.inputs = inputs
        self.attention_mask = attention_mask
        self.labels = labels
        # self.tokenizer = tokenizer
        # self.max_length = max_length

    # Return dataset size
    def __len__(self):
        return len(self.inputs)

    # Return a single example, including the id of the text and its corresponding label
    def __getitem__(self, index):
        # input_ids = self.tokenizer.encode(
        #     self.inputs[index], add_special_tokens=True, max_length=self.max_length
        # )
        # encoded_dict = self.tokenizer.encode_plus(
        #                 self.inputs[index],        # Sentence to encode
        #                 add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        #                 max_length = self.max_length, # Pad & truncate all sentences
        #                 pad_to_max_length = True,
        #                 return_attention_mask = True,   # Construct attn. masks
        #                 return_tensors = 'pt',     # Return pytorch tensors.
        #               )
        encoded_dict = {"input_ids": self.inputs[index], "attention_mask": self.attention_mask[index]}
        label = self.labels[index]
        # return input_ids, label
        return encoded_dict, label

# Definition of a model trainer for the bert-based sequence classification task
class BertModelTrainer:
    def __init__(self, train_file, val_file, test_file,
        model_name='bert-base-uncased', # Use the 12-layer BERT model, with an uncased vocab
        batch_size=16,  # author recommending: 16, 32
        num_epochs=4, # author recommending: 2, 3, 4
        learning_rate=2e-5, # 5e-5, 3e-5, 2e-5 for Adam
        warmup_steps=0.1,
    ):
        # Load the training, validation, and test data from JSON files
        self.train_data = self.load_data(train_file)
        self.val_data = self.load_data(val_file)
        self.test_data = self.load_data(test_file)
        self.max_length = max(
                            self.get_max_length(self.train_data),
                            self.get_max_length(self.val_data),
                            self.get_max_length(self.test_data)
                            )
        # Instantiate a tokenizer and a pre-trained model for sequence classification
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels = 2, # The number of output labels--2 for binary classification  
                        output_attentions = False, # do not return attentions weights
                        output_hidden_states = False, # do not return all hidden-states
                    ).cuda()
        # batch size & epoch number
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, eps = 1e-8)
        # Create the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(warmup_steps * len(self.train_data) / self.batch_size),
            num_training_steps=len(self.train_data) * self.num_epochs
        )

    # Load data from a JSON file and return a list of examples with labels ('0': human, '1': chatgpt)
    def load_data(self, filepath):
        with open(filepath, "r") as f:
          data = json.load(f)
        examples = []
        for example in data:
          # the example format: {"question": str, "text": str, "fake": int}
          text = example["text"]
          label = example["fake"]
          examples.append({"text": text, "label": label})
        return examples

    # Tokenize the inputs and labels and return them as two lists
    def tokenize_inputs(self, data):
        input_ids = []
        attention_masks = []
        labels = []
        # set max length according to data, 2 for [CLS] and [SEP]
        # max_length = max(len(example["text"]) for example in data) + 2
        for example in data:
            encoded_dict = self.tokenizer.encode_plus(
                        example["text"],           # Sentence to encode
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = self.max_length,   # Pad & truncate all sentences
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
            # Add the encoded sentence to the list
            input_ids.append(encoded_dict['input_ids'])
            # Add its attention mask to differentiate padding from non-padding
            attention_masks.append(encoded_dict['attention_mask'])
            # Add labels ('0':human or '1':chatgpt) to list
            labels.append(example["label"])
        # Convert the lists into pytorch tensors
        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(labels)
      
    def get_max_length(self, data):
        # set max length according to data, 2 for [CLS] and [SEP]
        return max(len(example["text"]) for example in data) + 2
    
    # Train the model on a given dataloader
    def train_model(self, train_loader):
        # Set the model to training mode and initialize the total loss
        self.model.train()
        total_loss = 0.0
        # tqdm package to display progress bar
        pbar = tqdm(total=len(train_loader))
        # Iterate over the batches in the dataloader
        for step, (encoded_dict, label) in enumerate(train_loader, start=1):
            # Clear the gradients, get the model outputs, and calculate the loss
            self.optimizer.zero_grad()
            outputs = self.model(
                encoded_dict['input_ids'].cuda(),
                attention_mask=encoded_dict['attention_mask'].cuda(),
            )
            loss = self.criterion(outputs.logits, label.cuda())
            # Backpropagate the loss, update the parameters, and adjust the learning rate
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # Update the total loss and calculate the average loss
            total_loss += loss.item()
            avg_loss = total_loss / step
            # Update the progress bar
            pbar.set_description(f"avg_loss: {avg_loss:.4f}")
            pbar.update(1)
        pbar.close()
        # Return the average loss over the entire dataset
        return total_loss / len(train_loader)

    # Evaluate the model on a given dataloader
    def evaluate_model(self, test_loader):
        # Set the model to evaluation mode and initialize the true and predicted labels
        self.model.eval()
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            # Iterate over the batches in the dataloader and get the model outputs
            for encoded_dict, labels in test_loader:
                outputs = self.model(
                    encoded_dict["input_ids"].cuda(),
                    attention_mask=encoded_dict["attention_mask"].cuda(),
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
    
    # Train and evaluate the model for a given number of epochs
    def run_training(self):
        # Tokenize the inputs and labels for the training, validation, and test datasets
        # Return torch tensor type
        train_inputs, train_masks, train_labels = self.tokenize_inputs(self.train_data)
        val_inputs, val_masks, val_labels = self.tokenize_inputs(self.val_data)
        test_inputs, test_masks, test_labels = self.tokenize_inputs(self.test_data)
        print("============= Finish Tokenization ================")
        # Create dataloaders for the training, validation, and test datasets
        train_dataset = CustomDataset(train_inputs, train_masks, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        print("============= Finish loading train dataset ================")
        val_dataset = CustomDataset(val_inputs, val_masks, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        print("============= Finish loading validate dataset ================")
        test_dataset = CustomDataset(test_inputs, test_masks, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        print("============= Finish loading test dataset ================")
        # Train the model for a given number of epochs and save the best model based on the validation accuracy
        best_accuracy = 0
        for epoch in range(self.num_epochs):
            train_loss = self.train_model(train_loader)
            val_report, val_accuracy = self.evaluate_model(val_loader)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), "best_model.pt")
            # Print the epoch number, training loss, and validation accuracy
            print(
                f"Epoch {epoch + 1}, train loss: {train_loss:.4f}, val accuracy: {val_accuracy:.4f}"
            )
            # Print the classification report for the validation dataset
            print(val_report)
        # Load the best model based on the validation accuracy and evaluate it on the test dataset
        self.model.load_state_dict(torch.load("best_model.pt"))
        test_report, test_accuracy = self.evaluate_model(test_loader)
        # Print the best accuracy and the classification report for the test dataset
        print(f"Best accuracy: {test_accuracy:.4f}")
        print(test_report)


if __name__ == "__main__":
    trainer = BertModelTrainer(
        "dataset/train.json", "dataset/val.json", "dataset/test.json"
    )
    trainer.run_training()