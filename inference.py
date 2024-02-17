# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/17 18:39
@Auth ： 吕鑫
@File ：inference.py
@IDE ：PyCharm
"""
import torch
from transformers import BertTokenizer
import json
from model import MultiLabelClassifier
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(args):
    """
    模型推理
    :param args:
    :return:
    """

    # 加载label2idx.json文件
    idx2label = json.load(open(args.idx2label_dir, "r"))
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    # 加载训练好的模型
    model = MultiLabelClassifier(args.tokenizer_path, args.output_dim, args.dropout, args.is_freeze_bert).to(device)
    model_state_dict = torch.load(args.model_path, map_location=device)["model_state_dict"]
    model.load_state_dict(model_state_dict)

    model.eval()  # 将模型设置为评估模式，关闭dropout和batch normalization等

    # 对输入文本进行tokenization
    inputs = tokenizer(args.text, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")

    # 获取模型预测结果
    with torch.no_grad():
        pred = model((inputs['input_ids'].to(device), inputs['attention_mask'].to(device)))

    probs = torch.sigmoid(pred)
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs.cpu() >= args.threshold)] = 1
    result = []
    for lst in y_pred:
        indices = [i for i, x in enumerate(lst) if x == 1]
        indices = [idx2label[str(i)] for i in indices]
        result.append(indices)

    return result


def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoint/fine_tuned_bert_model_9.pth")
    parser.add_argument("--tokenizer_path", type=str, default="/Users/lvxin/datasets/models/bert-base-uncased")
    parser.add_argument("--idx2label_dir", type=str, default="data/idx2label.json")

    parser.add_argument("--text", type=str, default="i am very angry now!")

    parser.add_argument("--output_dim", type=int, default=11)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--is_freeze_bert", type=bool, default=False)
    parser.add_argument("--max_length", type=int, default=128)

    return parser.parse_args()


def main():
    args = argument_parse()
    result_labels = inference(args)
    print(f"Predicted labels for the text: {args.text} are: {result_labels[0]}")


if __name__ == '__main__':
    main()
