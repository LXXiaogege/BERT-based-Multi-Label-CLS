# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/14 17:16
@Auth ： 吕鑫
@File ：data_preprocess.py
@IDE ：PyCharm
"""
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import json
from functools import partial


class MyDataset(Dataset):
    def __init__(self, data_path, sep, text_filed, idx2label_dir):
        self.data = pd.read_csv(data_path, sep=sep)

        self.text_filed = text_filed
        self.labels = list(json.load(open(idx2label_dir, "r")).values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data.loc[item][self.text_filed]
        label = self.data.loc[item][self.labels]  # .values.tolist()
        return text, label


def collate_fn(batch, tokenizer, max_length):
    """
    This function collates a batch of data and prepares it for input into a model.
    """
    batch_texts = [text for text, _ in batch]
    batch_labels = [label.astype(int).to_list() for _, label in batch]
    inputs = tokenizer(batch_texts, max_length=max_length, padding='max_length', truncation=True)
    inputs = {key: torch.tensor(value) for key, value in inputs.items()}
    batch_labels = torch.FloatTensor(batch_labels)

    return inputs, batch_labels


def load_dataset(data_path, tokenizer, args):
    dataset = MyDataset(data_path=data_path, sep=args.sep, text_filed=args.text_filed, idx2label_dir=args.idx2label_dir)
    my_partial_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, collate_fn=my_partial_collate_fn)
    return dataloader


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    data = pd.read_csv("data/dataset.csv", sep="\t")
    labels = list(json.load(open("data/idx2label.json", "r")).values())
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    train_data.to_csv("data/train.csv", sep="\t", index=False)
    test_data.to_csv("data/test.csv", sep="\t", index=False)
    pass
    # tokenizer = BertTokenizer.from_pretrained("/Users/lvxin/datasets/models/bert-base-uncased'")
    # load_dataset(data_path="dataset.csv", tokenizer=tokenizer, batch_size=8, shuffle=True)
