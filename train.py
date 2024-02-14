# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/14 17:28
@Auth ： 吕鑫
@File ：train.py
@IDE ：PyCharm
"""
from mydataset import load_dataset
from transformers import AutoModelForSequenceClassification
from model import MultiLabelClassifier
from torch.optim import Adam, AdamW
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # torch Dataset 对象
    train_loader, labels, label2idx, idx2label = load_dataset(data_path="dataset.csv", batch_size=batch_size,
                                                              shuffle=True)
    # model = AutoModelForSequenceClassification.from_pretrained(model_path,
    #                                                            problem_type="multi_label_classification",
    #                                                            num_labels=len(labels),
    #                                                            id2label=idx2label,
    #                                                            label2id=label2idx)
    model = MultiLabelClassifier(model_path=model_path, output_dim=len(labels), is_freeze_bert=False)

    lr = 2e-5
    num_epochs = 1
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.95)
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter()

    # 训练时的参数设置
    num_batches_show_loss = 50
    all_batch_count = 0
    total_loss = []
    # 训练模型
    for epoch in range(num_epochs):
        print("epoch", epoch)
        model.train()

        for batch in train_loader:
            all_batch_count += 1
            inputs,true_labels = batch
            pred = model(inputs)
            loss = criterion(pred, true_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            writer.add_scalar('Loss', loss.item(), all_batch_count)
            if all_batch_count % num_batches_show_loss == 0:
                writer.add_scalar('average total loss', np.mean(total_loss), all_batch_count)
                writer.add_scalar('latest average loss',
                                  np.mean(total_loss[-num_batches_show_loss:], all_batch_count % num_batches_show_loss))
                print(
                    f"current loss {loss.item():.4f}, average loss {np.mean(total_loss):.4f}, latest average loss: {np.mean(total_loss[-num_batches_show_loss:]):.4f}")

    pass


if __name__ == '__main__':
    model_path = "/Users/lvxin/datasets/models/bert-base-uncased"
    batch_size = 8
    train()
