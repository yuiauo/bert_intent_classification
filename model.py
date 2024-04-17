from transformers import BertModel
import torch
from typing import Final


MODEL: Final[str] = "DeepPavlov/rubert-base-cased"


class BertForIntentClassification(torch.nn.Module):
    """
        Русскоязычный БЕРТ с замороженными слоями + слой для классификации
        на num_classes классов
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self._bert_model = BertModel.from_pretrained(MODEL, add_pooling_layer=False)
        for param in self._bert_model.parameters():
            param.requires_grad = False

        self._classifier = torch.nn.Linear(768, num_classes)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self._bert_model(input_ids,  attention_mask=attention_mask)
        sequence_output = outputs[0][:, :1]
        logits = self._classifier(sequence_output.squeeze(0))
        return logits
