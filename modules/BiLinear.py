# -*- coding:utf-8 -*-

"""
@author : jfwang
@file : BiLinear.py
@time : 2020/11/10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np

class BiLinear(nn.Module):
    def __init__(self, left_features, right_features, out_features, bias=True):
        super(BiLinear, self).__init__()

        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.weight_left = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.weight_right = Parameter(torch.Tensor(self.out_features, self.right_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameter()

    def reset_paramteters(self):
        nn.init.xavier_uniform_(self.weight_left)
        nn.init.xavier_uniform_(self.weight_right)
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        batch_size = input_left.size()[:-1]
        batch = int(np.prod(batch_size))

        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(batch, self.left_features)
        input_right = input_right.view(batch, self.right_features)

        # output [batch, out_features]
        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.weight_left, None) + F.linear(input_right, self.weight_right, None)
        # convert back to [batch1, batch2, ..., out_features]
        return output.view(batch_size + (self.out_features,))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'left_features=' + str(self.left_features) \
               + ', right_features=' + str(self.right_features) \
               + ', out_features=' + str(self.out_features) + ')'