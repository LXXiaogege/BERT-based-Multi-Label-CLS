# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/14 17:28
@Auth ： 吕鑫
@File ：train.py
@IDE ：PyCharm
"""
from data_preprocess import load_dataset
from transformers import BertTokenizer
from model import MultiLabelClassifier
from torch.optim import AdamW
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, zero_one_loss, hamming_loss, precision_score
import json
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    """
    训练函数，用于训练多标签分类模型
    :param args: 命令行参数
    """
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    train_loader = load_dataset(data_path=args.train_dir, tokenizer=tokenizer, args=args)

    idx2label = json.load(open(args.idx2label_dir, "r"))
    # label2idx = {v: k for k, v in idx2label.items()}
    labels = list(idx2label.values())

    model = MultiLabelClassifier(model_path=args.model_path, output_dim=len(labels), dropout=args.dropout,
                                 is_freeze_bert=args.is_freeze_bert)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    # sigmoid + BCE Loss Function
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter()
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

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
            pred = model((inputs['input_ids'].to(device), inputs['attention_mask'].to(device)))
            loss = criterion(pred, true_labels.to(device))
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
            evaluate(args, model, tokenizer, target_names=labels)
        # 每一轮次测试一次
        evaluate(args, model, tokenizer, target_names=labels)

        try:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                }, f"checkpoint/fine_tuned_bert_model_{epoch}.pth")
        except OSError as error:
            print(f"OS error: {error}")


def evaluate(args, model, tokenizer, target_names):
    """
    在测试集上评估模型
    """
    print("*********evaluate*********")
    model.eval()
    test_loader = load_dataset(data_path=args.test_dir, tokenizer=tokenizer, args=args)
    pred_labels = []
    true_labels = []
    for batch in test_loader:
        inputs, true_label = batch
        pred = model((inputs['input_ids'].to(device), inputs['attention_mask'].to(device)))
        probs = torch.sigmoid(pred)
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs.cpu() >= args.threshold)] = 1

        true_labels.extend(true_label.tolist())
        pred_labels.extend(y_pred)

    f1_average = f1_score(y_true=true_labels, y_pred=pred_labels, average='samples', zero_division=0)
    roc_auc = roc_auc_score(true_labels, pred_labels, average='samples',labels=range(len(target_names)))
    accuracy = accuracy_score(true_labels, pred_labels, normalize=True, sample_weight=None)
    precision = precision_score(y_true=true_labels, y_pred=pred_labels, average='samples', zero_division=0)
    hamming = hamming_loss(y_true=true_labels, y_pred=pred_labels)  # hamming loss
    zero_one = zero_one_loss(y_true=true_labels, y_pred=pred_labels)  # 0-1 loss

    report = classification_report(true_labels, pred_labels, zero_division=0, target_names=target_names,
                                   labels=range(len(target_names)))
    confusion_matrix = multilabel_confusion_matrix(y_true=true_labels, y_pred=pred_labels,
                                                   labels=range(len(target_names)))

    print(f"hamming loss: {hamming:.4f}, zero_one_loss: {zero_one:.4f}")
    print(f"f1 score: {f1_average:.4f}, roc_auc: {roc_auc:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f}")
    print("classification report:", report)
    print("confusion_matrix:", confusion_matrix)


def argument_parse():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    # 数据文件参数
    parser.add_argument("--model_path", type=str, default="/Users/lvxin/datasets/models/bert-base-uncased")
    parser.add_argument("--train_dir", type=str, default="data/train.csv")
    parser.add_argument("--test_dir", type=str, default="data/test_1.csv")
    parser.add_argument("--idx2label_dir", type=str, default="data/idx2label.json")
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--text_filed", type=str, default="Tweet")

    # 模型训练参数
    parser.add_argument("--shuffle", type=bool, default=True)  # 是否打乱训练集
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_batches_show_loss", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.95)

    # 评价指标阈值设置
    parser.add_argument("--threshold", type=float, default=0.5)

    # 模型参数设置
    # parser.add_argument("--output_dim", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--is_freeze_bert", type=bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = argument_parse()
    train(args=args)


if __name__ == '__main__':
    main()
