import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

class BERTBase(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased",
            config=BertConfig.from_pretrained(
                "bert-base-cased",
                num_labels=num_labels,
            )
        )
        trainable_layers = [self.model.bert.encoder.layer[-1], self.model.bert.pooler, self.model.classifier]
        total_params = 0
        trainable_params = 0
        for p in self.model.parameters():
                p.requires_grad = False
                total_params += p.numel()

        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
                trainable_params += p.numel()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def name(self):
        return "bert-base-cased"