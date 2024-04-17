#!/usr/bin/python3.11

from model import BertForIntentClassification
from read_data import get_intents
from data import Intent, IntentModel
from utils import dataloader, process, tokenize
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import logging, AdamW
from typing import Callable
from warnings import filterwarnings
from sklearn.metrics import f1_score


logging.set_verbosity_error()
filterwarnings("ignore")

PATH = "my_model5.pt"


class Model:

    def __init__(self, intents: list[IntentModel]):
        self.model = BertForIntentClassification(num_classes=len(intents))
        self._init_attrs(intents)
        self.model.to(self.device)

    def _compose_label_dict(self, intents: list[IntentModel]) -> dict[int, Intent]:
        label = 0
        label_dict = {}
        for intent in intents:
            label_dict[label] = intent.tag
            label += 1
        return label_dict

    def _init_attrs(self, intents: list[IntentModel]) -> None:
        input_ids, attention_mask, labels = process(intents)
        self.label_dict = self._compose_label_dict(intents)
        self.dataloader = dataloader(input_ids, attention_mask, labels)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3)
        self.loss = self._loss_func(labels)
        self.epochs = 50

    def _loss_func(
            self,
            labels: torch.Tensor
    ) -> Callable[[torch.Tensor, torch.Tensor], ...]:
        class_wts = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels.numpy()),
            y=labels.numpy())
        weight = torch.tensor(class_wts, dtype=torch.float)
        weight = weight.to(self.device)
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
        return cross_entropy

    def save(self) -> None:
        torch.save(self.model.state_dict(), PATH)

    def get(self) -> None:
        model = BertForIntentClassification(2)
        model.load_state_dict(torch.load(PATH))
        model.eval()

    def predict(self, input_text: str) -> np.ndarray[float, float]:
        input_ids, attention_mask = tokenize(input_text)
        with torch.no_grad():
            predictions = self.model(
                input_ids.to(self.device),
                attention_mask.to(self.device)
            )
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            predictions = predictions.detach().cpu().numpy()

        return predictions

    def _batch_train(self) -> tuple[float, float]:
        total_loss, total_accuracy = 0, 0
        total_preds = []
        total_labels = []

        for step, batch in enumerate(self.dataloader):
            if step % 100 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.dataloader)))
            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch
            self.model.zero_grad()
            preds = self.model(sent_id, mask)
            loss = self.loss(preds, labels)
            total_loss = total_loss + loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            total_preds += list(preds)
            total_labels += labels.tolist()
        avg_loss = total_loss / len(self.dataloader)

        f1 = f1_score(total_labels, total_preds, average='weighted')
        return avg_loss, f1

    def train(self) -> None:
        self.model.train()
        train_losses = []
        for epoch in range(self.epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
            train_loss, f1_train = self._batch_train()
            train_losses.append(train_loss)
            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'\nTraining F1: {f1_train:.3f}')


# if __name__ == "__main__":

    # intents = get_intents()
    #
    # m = Model(intents)
    # m.train()
    #
    # m.save()
    # print("SAVED")
    # raise Exception
    #
    #
    # # m.get(PATH)
    # print("LOADED")
    # all_results = []
    #
    # for j in (
    #   "ку", "привет", "здорова", "пока", "покеда", "гудбай"
    # ):
    #     p = m.predict(j)
    #     label = np.argmax(p, axis=1)[0]
    #     result: float = p[0][label]
    #     intent: Intent = m.label_dict[label]
    #     print(j, "|", label, "|", result, "|", intent)
    #     print("==========================")
