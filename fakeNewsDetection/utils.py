import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from summarizer import Summarizer
from summarizer.coreference_handler import CoreferenceHandler
from transformers import (BartForConditionalGeneration,
                          BartTokenizerFast, BertForSequenceClassification,
                          DistilBertConfig, DistilBertModel, DistilBertTokenizerFast)

sum_tokenizer = BartTokenizerFast.from_pretrained(
    'sshleifer/distilbart-cnn-12-6')
sum_model = BartForConditionalGeneration.from_pretrained(
    'sshleifer/distilbart-cnn-12-6')

custom_config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
custom_config.output_hidden_states = True
custom_tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased")
custom_model = DistilBertModel.from_pretrained(
    "distilbert-base-uncased", config=custom_config)
handler = CoreferenceHandler("en_core_web_sm")


def convert_to_int(rating):
    if(rating == 'TRUE' or rating == "true" or rating == True):
        return 0
    if(rating == 'FALSE' or rating == "false" or rating == False):
        return 1
    if(rating == "partially false"):
        return 2
    else:
        return 3


def convert_to_rating(int):
    if(int == 0):
        return "true"
    if(int == 1):
        return "false"
    if(int == 2):
        return "partially false"
    else:
        return "other"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def get_extractive_summary(extractive_sum_model, text):
    sum_text = extractive_sum_model(text, ratio=0.4)
    if not sum_text:
        return text
    else:
        return sum_text


def generate_extractive_summaries(dataframe, filename):
    extractive_sum_model = Summarizer(
        custom_model=custom_model, custom_tokenizer=custom_tokenizer, sentence_handler=handler, random_state=43)

    dataframe['text_extractive'] = dataframe.apply(
        lambda x: get_extractive_summary(extractive_sum_model, x['text']), axis=1)
    dataframe = dataframe[["public_id", "title",
                           "text", "text_extractive", "our rating"]]
    dataframe.to_csv(filename, index=False)


def get_abstractive_summary(text):
    summary = ""
    # Split longer texts into documents with overlapping parts
    if(len(text.split()) > 1000):
        texts = get_split(text, 1000)
        inputs = sum_tokenizer(texts, return_tensors='pt', truncation="only_first",
                               padding="max_length", max_length=1024).input_ids
        # Generate Summary
        output = sum_model.generate(
            inputs, min_length=400, max_length=512, top_k=100, top_p=.95, do_sample=True)
        sum_texts = sum_tokenizer.batch_decode(
            output, skip_special_tokens=True)
        summary = "".join(sum_texts)
    else:
        inputs = sum_tokenizer(
            text, return_tensors='pt', truncation=True, padding=True, max_length=1024).input_ids
        # Generate Summary
        output = sum_model.generate(inputs, min_length=int(
            len(text.split())*0.4), max_length=512, top_k=100, top_p=.95, do_sample=True)
        sum_texts = sum_tokenizer.batch_decode(
            output, skip_special_tokens=True)
        summary = sum_texts[0]
    return summary

def generate_abstractive_summaries(dataframe, filename):
    dataframe['text_abstractive'] = dataframe.apply(
        lambda x: get_abstractive_summary(x['text']), axis=1)
    dataframe = dataframe[["public_id", "title", "text",
                           "text_extractive", "text_abstractive", "our rating"]]
    dataframe.to_csv(filename, index=False)


def get_encodings_test(dataframe, tokenizer, summary=0):
    encodings = []
    for idx in range(len(dataframe)):
        if(summary == 2):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text_extractive']
        if(summary == 1):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text_abstractive']
        else:
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text']
        # Split longer texts into documents with overlapping parts
        if(len(sum_text.split()) > 500):
            text_parts = get_split(sum_text, 500)
            tensors = tokenizer(
                text_parts, padding="max_length", truncation="only_first")
            # Dimensional mean of the tensor to represent all parts of the text
            mean_input_ids = list(np.mean(tensors.input_ids, axis=0))
            mean_attention_mask = list(np.mean(tensors.attention_mask, axis=0))
            tensors.data['input_ids'] = mean_input_ids
            tensors.data['attention_mask'] = mean_attention_mask
            encodings.append(tensors)
        else:
            encodings.append(
                tokenizer(sum_text, padding="max_length", truncation="only_first"))

    return encodings


def get_encodings(dataframe, tokenizer, summary=0):
    encodings = []
    labels = []
    for idx in range(len(dataframe)):
        if(summary == 2):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text_extractive']
        if(summary == 1):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text_abstractive']
        elif(summary == 0):
            sum_text = str(dataframe.iloc[idx]['title']) + \
                ". " + dataframe.iloc[idx]['text']

        # Split longer texts into documents with overlapping parts
        if(len(sum_text.split()) > 500):
            text_parts = get_split(sum_text, 500)
            tensors = tokenizer(
                text_parts, padding="max_length", truncation="only_first")
            # Dimensional mean of the tensor to represent all parts of the text
            mean_input_ids = list(np.mean(tensors.input_ids, axis=0))
            mean_attention_mask = list(np.mean(tensors.attention_mask, axis=0))
            tensors.data['input_ids'] = mean_input_ids
            tensors.data['attention_mask'] = mean_attention_mask
            encodings.append(tensors)
            labels.append(dataframe.iloc[idx]['label'])
        else:
            encodings.append(
                tokenizer(sum_text, padding="max_length", truncation="only_first"))
            labels.append(dataframe.iloc[idx]['label'])


    return encodings, labels

# This class represent one part of the dataset (either training, validation or test via subclass)
class CheckThatLabDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        tmp_list = []
        for key in self.encodings[idx]['input_ids']:
            # Round the values to the next integer as double values are not allowed here
            tmp_list.append(int(round(key)))

        item['input_ids'] = torch.tensor(tmp_list)
        tmp_list2 = []
        for key in self.encodings[idx]['attention_mask']:
            tmp_list2.append(key)

        item['attention_mask'] = torch.tensor(tmp_list2)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class CheckThatLabDatasetTest(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {}
        tmp_list = []
        for key in self.encodings[idx]['input_ids']:
            tmp_list.append(int(round(key)))

        item['input_ids'] = torch.tensor(tmp_list)
        tmp_list2 = []
        for key in self.encodings[idx]['attention_mask']:
            tmp_list2.append(key)

        item['attention_mask'] = torch.tensor(tmp_list2)

        return item

    def __len__(self):
        return len(self.encodings)

def get_split(text, split_length, stride_length=50):
    l_total = []
    l_partial = []
    text_length = len(text.split())
    partial_length = split_length - stride_length
    if text_length//partial_length > 0:
        n = text_length//partial_length
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_partial = text.split()[:split_length]
            l_total.append(" ".join(l_partial))
        else:
            l_partial = text.split()[w*partial_length:w *
                                     partial_length + split_length]
            l_total.append(" ".join(l_partial))
    return l_total


def init_full_text_model():
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)