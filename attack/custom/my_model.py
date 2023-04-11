# my_model.py file to load custom model
import torch
import textattack
from transformers import BertTokenizer, BertForSequenceClassification
# from textattack.models.wrappers import PyTorchModelWrapper

state_dict_load = torch.load("/content/drive/MyDrive/Colab Notebooks/repos/Bert-based-Text-Detection/best_model_v2.pt")
best_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")# replace this line with your model loading code
best_model.load_state_dict(state_dict_load)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # replace this line with your tokenizer loading code
# model = PyTorchModelWrapper(best_model, tokenizer)
model = textattack.models.wrappers.HuggingFaceModelWrapper(best_model, tokenizer)