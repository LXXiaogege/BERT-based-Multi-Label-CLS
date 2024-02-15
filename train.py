# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/14 17:28
@Auth ： 吕鑫
@File ：train.py
@IDE ：PyCharm
"""
from mydataset import load_dataset
from transformers import AutoModelForSequenceClassification, BertTokenizer
from model import MultiLabelClassifier
from torch.optim import Adam, AdamW
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    train_loader = load_dataset(data_path=args.train_dir, tokenizer=tokenizer, batch_size=args.batch_size, shuffle=True)

    idx2label = json.load(open(args.idx2label, "r"))
    # label2idx = {v: k for k, v in idx2label.items()}
    labels = list(idx2label.values())

    model = MultiLabelClassifier(model_path=args.model_path, output_dim=len(labels), is_freeze_bert=args.is_freeze_bert)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.95)
    # sigmoid + BCE Loss Function
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter()

    # 训练时的参数设置
    num_batches_show_loss = args.num_batches_show_loss
    all_batch_count = 0
    total_loss = []
    # 训练模型
    for epoch in range(args.epochs):
        print("epoch", epoch)
        model.train()

        for batch in train_loader:
            all_batch_count += 1
            inputs, true_labels = batch
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
        # 每一轮次测试一次
        eval(args, model, tokenizer, idx2label, len(labels), args.batch_size)


def eval(args, model, tokenizer, idx2label, class_nums, batch_size):
    print("*********evaluate*********")
    model.eval()
    test_loader = load_dataset(data_path=args.test_dir, tokenizer=tokenizer, batch_size=batch_size, shuffle=False)
    pred_labels = []
    true_labels = []
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        pred = model(inputs)
        _, y_pred = torch.max(pred, 1)
        pred_labels.extend(y_pred.cpu().numpy())
        true_labels.extend(inputs['labels'].cpu().numpy())

    class_names = list(idx2label.values())
    report = classification_report(true_labels, pred_labels, target_names=class_names,
                                   labels=range(class_nums))
    print("Classification Report", report)
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=range(class_nums))
    print("Confusion Matrix", conf_matrix)


def argument_parse():
    parser = argparse.ArgumentParser()
    # 文件路径
    parser.add_argument("--model_path", type=str, default="/Users/lvxin/datasets/models/bert-base-uncased")
    parser.add_argument("--train_dir", type=str, default="train.csv")
    parser.add_argument("--test_dir", type=str, default="test.csv")
    parser.add_argument("--idx2label", type=str, default="idx2label.json")

    # 模型训练参数
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_batches_show_loss", type=int, default=10)

    parser.add_argument("--output_dim", type=int, default=10)
    parser.add_argument("--is_freeze_bert", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = argument_parse()
    train(args=args)


if __name__ == '__main__':
    main()
