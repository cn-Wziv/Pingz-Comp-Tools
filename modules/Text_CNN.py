# -*- coding:utf-8 -*-

"""
@author : jfwang
@file : Text_CNN.py
@time : 2020/11/10
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

'''
text cnn
可使用预训练词向量, 或拼接到bert之后
'''

class CNN_Text(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num, kernel_sizes, dropout):
        super(CNN_Text, self).__init__()
        # self.args = args

        # V = embed_num
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes

        # self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)

        logit = self.fc1(x)  # (N, C)

        return logit