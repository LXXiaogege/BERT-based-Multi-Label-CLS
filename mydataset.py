# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/14 17:16
@Auth ： 吕鑫
@File ：mydataset.py
@IDE ：PyCharm
"""
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
import torch
import json
from functools import partial


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, sep="\t")

        self.text_filed = "Tweet"
        self.labels = list(json.load(open("idx2label.json", "r")).values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data.loc[item][self.text_filed]
        label = self.data.loc[item][self.labels]  # .values.tolist()
        return text, label


def collate_fn(batch, tokenizer):
    """
    collate_fn is used to merge a list of samples to form a mini-batch.
    """
    batch_texts = [text for text, _ in batch]
    batch_labels = [label.astype(int).to_list() for _, label in batch]
    inputs = tokenizer(batch_texts, max_length=512, padding='max_length', truncation=True)
    inputs = {key: torch.tensor(value) for key, value in inputs.items()}
    batch_labels = torch.FloatTensor(batch_labels)

    return inputs, batch_labels


def load_dataset(data_path, tokenizer, batch_size=8, shuffle=True):
    dataset = MyDataset(data_path=data_path)
    my_partial_collate_fn = partial(collate_fn, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_partial_collate_fn)
    return dataloader


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    data = pd.read_csv("dataset.csv", sep="\t")
    labels = list(json.load(open("idx2label.json", "r")).values())
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    train_data.to_csv("train.csv", sep="\t", index=False)
    test_data.to_csv("test.csv", sep="\t", index=False)
    pass
    # tokenizer = BertTokenizer.from_pretrained("/Users/lvxin/datasets/models/bert-base-uncased'")
    # load_dataset(data_path="dataset.csv", tokenizer=tokenizer, batch_size=8, shuffle=True)
