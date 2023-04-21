# inherit to create a new class
import torch
from torch.nn import CrossEntropyLoss
import textattack
from textattack.models.wrappers import PyTorchModelWrapper

# torch.cuda.empty_cache()

class BertDetectorModelWrapper(PyTorchModelWrapper):
    """Loads a PyTorch-model-based finetuned BERT model (`nn.Module`) and BERTtokenizer.
    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: BERTtokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list, batch_size=32):
        model_device = next(self.model.parameters()).device
        # ids = self.tokenizer(text_input_list)
        input_ids = self.tokenizer.encode(
                text_input_list,
                add_special_tokens=True,
                truncation=True,
                max_length=512,
            )
        ids = torch.tensor(ids).to(model_device)

        with torch.no_grad():
            outputs = self.model(
                    inputs["input_ids"].cuda(),
                    attention_mask=inputs["attention_mask"].cuda(),
                )

        return outputs