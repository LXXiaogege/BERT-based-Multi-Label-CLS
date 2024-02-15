# -*- coding: utf-8 -*-
"""
@Time ： 2024/2/11 12:19
@Auth ： 吕鑫
@File ：model.py
@IDE ：PyCharm
"""
from torch import nn
import torch
from transformers import BertModel


class MultiLabelClassifier(nn.Module):
    def __init__(self, model_path, output_dim, dropout=0.1, is_freeze_bert=False):
        super(MultiLabelClassifier, self).__init__()
        self.output_dim = output_dim
        self.is_freeze_bert = is_freeze_bert
        self.bert_model = BertModel.from_pretrained(model_path)
        self.input_dim = self.bert_model.config.hidden_size
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.fc = nn.Linear(self.input_dim, self.output_dim, bias=True)

    def forward(self, x):
        if self.is_freeze_bert:
            with torch.no_grad():
                bert_outputs = self.bert_model(x['input_ids'], attention_mask=x['attention_mask'])
        else:
            bert_outputs = self.bert_model(x['input_ids'], attention_mask=x['attention_mask'])
        pooled_output = bert_outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.fc(pooled_output)
        return outputs
