import torch
import torch.nn as nn
from transformers import AutoModel

class ReviewClassifier(nn.Module):
    def __init__(self, model_path='distilbert-base-uncased'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)