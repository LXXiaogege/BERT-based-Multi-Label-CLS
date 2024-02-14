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

tokenizer = BertTokenizer.from_pretrained('/Users/lvxin/datasets/models/bert-base-uncased')


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, sep="\t")

        self.text_filed = "Tweet"
        # label2idx
        self.labels = self.data.columns[2:].tolist()
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data.loc[item][self.text_filed]
        label = self.data.loc[item][self.labels]  # .values.tolist()
        return text, label


def collate_fn(batch):
    """
    collate_fn is used to merge a list of samples to form a mini-batch.
    """
    batch_texts = [text for text, _ in batch]
    batch_labels = [label.astype(int).to_list() for _, label in batch]
    inputs = tokenizer(batch_texts, max_length=512, padding='max_length', truncation=True)
    inputs = {key: torch.tensor(value) for key, value in inputs.items()}
    batch_labels = torch.FloatTensor(batch_labels)

    return inputs, batch_labels


def load_dataset(data_path, batch_size=8, shuffle=True):
    dataset = MyDataset(data_path=data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader, dataset.labels, dataset.label2idx, dataset.idx2label


if __name__ == '__main__':
    load_dataset(data_path="dataset.csv", batch_size=8, shuffle=True)
