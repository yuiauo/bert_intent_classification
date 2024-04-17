from collections.abc import Iterator
from model import BertForIntentClassification
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import logging, AdamW, BertTokenizer
from typing import Callable
from warnings import filterwarnings
from sklearn.metrics import f1_score
from typing import Final, Literal, Protocol
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from data import IntentModel, Intent


TOKENIZER: Final[str] = "DeepPavlov/rubert-base-cased"
BATCH_SIZE: Final[int] = 1
MAX_LENGTH: Final[int] = 64
EPOCHS: Final[int] = 64

logging.set_verbosity_error()
filterwarnings("ignore")
tokenizer = BertTokenizer.from_pretrained(TOKENIZER)

DEVICE = Literal["cuda", "cpu"]
SAVE_PATH: Final[str] = 'topic_saved_weights.pt'


class Model(Protocol):
    """   Достаточный протокол модели Pytorch, остальное можно изучить здесь:
        torch.nn.Module """
    intents: dict[int, Intent]

    def eval(self) -> None:
        """ Перевод модели в "предсказательское" состояние """
        ...

    def train(self) -> None:
        """ Перевод модели в состояние при котором проходит обучение """
        ...

    def to(self, device: DEVICE) -> None:
        """ Явное указание устройства на котором будут производиться вычисления """
        ...

    def forward(self, *args, **kwargs) -> torch.Tensor:
        ...

    # def __call__(self, *args, **kwargs) -> torch.Tensor:
    #     ...
    #     logits = self.forward(*args, **kwargs)
    #     ...
    #     return logits

    def zero_grad(self) -> None:
        """ Явное обнуление градиента. Если не производить, текущий градиент
            будет суммой всех предыдущих """
        ...

    def parameters(self) -> Iterator:
        """ Возвращает параметры модели """
        ...


class Pipeline:

    @staticmethod
    def save_model(model: Model, path: str) -> None:
        state = {
            'model_intents': model.intents,
            'model': model
        }
        torch.save(state, path)

    @staticmethod
    def get_model(path: str) -> Model:
        state = torch.load(path)
        model_intents = state.get("model_intents")
        actual_intents = Pipeline.get_actual_intents()
        if model_intents != actual_intents:
            raise Exception("Модель нужно переобучить!")
        return state.get("model")

    @staticmethod
    def get_actual_intents() -> list[IntentModel]:
        """ Получение интентов из БД """
        ...

    @staticmethod
    def intents_to_tensors(intents: list[IntentModel]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[int, Intent]
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
                single_input_id, single_attention_mask = Pipeline.tokenize(example)
                input_ids.append(single_input_id)
                attention_mask.append(single_attention_mask)
                y.append(label)
            label_dict[label] = intent.tag
            label += 1

        return (
            torch.stack(input_ids).squeeze(1),
            torch.stack(attention_mask).squeeze(1),
            torch.tensor(y),
            label_dict
        )

    @staticmethod
    def dataloader(input_ids, attention_mask, labels):
        dataset = TensorDataset(
            input_ids, attention_mask, labels
        )
        sampler = RandomSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

    @staticmethod
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

    @staticmethod
    def train(model_or_path: Model | str) -> None:
        match model_or_path:
            case str():
                model = Pipeline.get_model(model_or_path)
            case _:
                model = model_or_path
        model.train()
        train_losses = []
        for epoch in range(EPOCHS):
            print('\n Epoch {:} / {:}'.format(epoch + 1, EPOCHS))
            train_loss, f1_train = Pipeline.batch_train()
            train_losses.append(train_loss)
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'\nTraining F1: {f1_train:.3f}')

    @staticmethod
    def loss_func(labels: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor], ...]:
        class_wts = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels.numpy()),
            y=labels.numpy())
        weight = torch.tensor(class_wts, dtype=torch.float)
        weight = weight.to(DEVICE)
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
        return cross_entropy

    @staticmethod
    def batch_train(model: Model, dataloader: DataLoader) -> tuple[float, float]:
        total_loss, total_accuracy = 0, 0

        total_preds = []
        total_labels = []

        for step, batch in enumerate(dataloader):
            if step % 10 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))
            batch = [r.to(DEVICE) for r in batch]
            sent_id, mask, labels = batch
            model.zero_grad()
            preds = model(sent_id, mask)
            loss = self.loss(preds, labels)
            total_loss = total_loss + loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()
            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            total_preds += list(preds)
            total_labels += labels.tolist()
        avg_loss = total_loss / len(self.dataloader)
        print(total_loss)

        f1 = f1_score(total_labels, total_preds, average='weighted')
        return avg_loss, f1

    def __init__(self, model:):

        self.model = BertForIntentClassification(num_classes=len(intents))
        self._init_attrs(intents)
        self.model.to(self.device)

    def __init__(self, intents: list[IntentModel]):

        self.model = BertForIntentClassification(num_classes=len(intents))
        self._init_attrs(intents)
        self.model.to(self.device)

    def _init_attrs(self, intents: list[IntentModel]) -> None:

        input_ids, attention_mask, labels, label_dict = Pipeline.intents_to_tensors(
            intents)

        self.dataloader = Pipeline.dataloader(input_ids, attention_mask, labels)
        self.label_dict = label_dict
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3)
        self.loss = self._loss_func(labels)
        self.epochs = 120

    def predict(self, input_text: str) -> np.ndarray[int]:
        input_ids, attention_mask = tokenize(input_text)
        with torch.no_grad():
            predictions = self.model(
                input_ids.to(self.device),
                attention_mask.to(self.device)
            )
            predictions = predictions.detach().cpu().numpy()
        return np.argmax(predictions, axis=1)


# intent1 = IntentModel(tag="hi", examples=[
#     "привет",
#     "здарова"
# ])
# intent2 = IntentModel(tag="by", examples=[
#     "пока",
#     "бай"
# ])
#
# intents = [intent1, intent2]
#
# m = Model(intents)
# m.train()
#
# res = m.predict("пока")
#
# print("+++++", res)
