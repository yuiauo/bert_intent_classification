import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import BertTokenizer
from typing import Final

from data import IntentModel


TOKENIZER: Final[str] = "DeepPavlov/rubert-base-cased"
BATCH_SIZE: Final[int] = 1
MAX_LENGTH: Final[int] = 32


tokenizer = BertTokenizer.from_pretrained(TOKENIZER)


def tokenize(input_text: str) -> tuple[torch.Tensor, torch.Tensor]:
    tokenized = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_token_type_ids=False
    )
    return tokenized["input_ids"], tokenized["attention_mask"]


def process(intents: list[IntentModel]) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    :returns  1) torch.Tensor, тензор содержащий векторные представления всех
     examples переданных для обучения, размер len(examples) X len(
    """
    y = []
    input_ids = []
    attention_mask = []
    label_dict = {}

    label = 0
    for intent in intents:
        for example in intent.examples:
            single_input_id, single_attention_mask = tokenize(example)
            input_ids.append(single_input_id)
            attention_mask.append(single_attention_mask)
            y.append(label)
        label_dict[label] = intent.tag
        label += 1

    return (
        torch.stack(input_ids).squeeze(1),
        torch.stack(attention_mask).squeeze(1),
        torch.tensor(y)
    )


def dataloader(input_ids, attention_mask, labels):

    dataset = TensorDataset(
        input_ids, attention_mask, labels
    )

    sampler = RandomSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)
