import json
import codecs

from data import IntentModel

fp = codecs.open("intents_mts_bank_tm_7009_mal-13854-Iuo.json", "r", "utf_8_sig")

# intent_dict = json.load(fp)

#
# def get_intents():
#     all_intents = []
#
#     for intent in intent_dict:
#         intent_tag = intent["path"].replace("/", "")
#         intent_examples = [phrase["text"] for phrase in intent["phrases"]]
#
#         if intent_tag and intent_examples:
#             intnt = IntentModel(tag=intent_tag, examples=intent_examples)
#             all_intents.append(intnt)
#
#     return all_intents


def get_intents():

    def get_robot():
        with codecs.open("Робот.txt", "r", "utf_8_sig") as robot:
            intent = "Робот"
            examples = []
            for line in robot:
                examples.append(line.strip())
            examples.append(intent)
        return IntentModel(tag=intent, examples=examples)

    def get_refuse():
        with codecs.open("Четкий отказ.txt", "r", "utf_8_sig") as robot:
            intent = "Четкий отказ"
            examples = []
            for line in robot:
                examples.append(line.strip())
            examples.append(intent)
        return IntentModel(tag=intent, examples=examples)

    def get_auto():
        with codecs.open("Автоответчик.txt", "r", "utf_8_sig") as robot:
            intent = "Автоответчик"
            examples = []
            for line in robot:
                examples.append(line.strip())
            examples.append(intent)
        return IntentModel(tag=intent, examples=examples)

    def get_yes():
        with codecs.open("Да.txt", "r", "utf_8_sig") as robot:
            intent = "Да"
            examples = []
            for line in robot:
                examples.append(line.strip())
            examples.append(intent)
        return IntentModel(tag=intent, examples=examples)

    def get_operator():
        with codecs.open("Оператор.txt", "r", "utf_8_sig") as robot:
            intent = "Оператор"
            examples = []
            for line in robot:
                examples.append(line.strip())
            examples.append(intent)
        return IntentModel(tag=intent, examples=examples)

    def get_rep():
        with codecs.open("Повторное приветствие.txt", "r", "utf_8_sig") as robot:
            intent = "Повторное приветствие"
            examples = []
            for line in robot:
                examples.append(line.strip())
            examples.append(intent)
        return IntentModel(tag=intent, examples=examples)

    return [
        get_rep(), get_refuse(), get_robot(), get_auto(), get_yes(), get_operator()
    ]

