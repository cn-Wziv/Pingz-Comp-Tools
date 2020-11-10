# -*- coding:utf-8 -*-

"""
@author : jfwang
@file : adversarial_training.py
@time : 2020/11/10
"""

'''
两种对抗训练的方式: FGM & PGD
加入在embeding层的扰动
实现及使用demo如下
'''

import torch


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='emb.'):
        # emb_name 要替换为模型中的embedding层的名字
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if notm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)


    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# 使用以下代码进行使用
fgm = FGM(model)
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward()

    # 对抗训练
    fgm.attack()  # 在embedding上添加对抗扰动
    loss_adv = model(batch_input, batch_loss_label)
    loss_adv.backward()
    fgm.restore()  # 恢复embedding层的参数

    # 梯度下降, 更新参数
    optimizer.step()
    model.zero_grad()

import torch


# ===================================================================
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb.', is_first_attack = False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)


    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}


    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r


    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()


    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


# 使用以下代码调用
# bert_model.embeddings.word_embeddings.weight
pgd = PGD(model)
K = 3
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward()
    pgd.backup_grad()

    # 对抗训练
    for t in range(K):
        pgd.attack(is_first_attack(t == 0))  # 在第一次attack时备份
        if t != K - 1:
            model.zero_grad()
        else:
            pgd.restore_grad()

        loss_adv = model(batch_input, batch_label)
        loss_adv.backward()
    pgd.restore()

    # 梯度下降, gentian参数
    optimizer.step()
    model.zero_grad()
