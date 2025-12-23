import math

import torch
import torch.nn as nn


class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()

        # 计算卷积核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 计算padding的值
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        # nn.Conv2d初始化
        for m in self.modules():  # self.modules()包含网络模块的自己本身和所有后代模块。
            if isinstance(m, nn.Conv1d):  # isinstance()是Python中的一个内建函数。是用来判断一个对象的变量类型。
                nn.init.kaiming_normal_(m.weight)  # 权重初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置项初始化

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


class MyMultiheadAttentionCon(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(MyMultiheadAttentionCon, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.head_size = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        # Linear transformations
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # 计算注意力权重张量 单头的输出: torch.Size([4, 5, 5])
        scores = torch.reciprocal(torch.matmul(query, key.transpose(-2, -1))) / (self.head_size ** 0.5)
        # scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn_output_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_output_weights, value).transpose(1, 2).contiguous().view(batch_size, -1,
                                                                                                 self.d_model)
        attn_output_weights = torch.mean(attn_output_weights, dim=1)
        return attn_output, attn_output_weights


class MyMultiheadAttentionTanhRelu(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(MyMultiheadAttentionTanhRelu, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.head_size = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, query, key, value):
        batch_size = query.size(0)
        # Linear transformations
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # 计算注意力权重张量 单头的输出: torch.Size([4, 5, 5])
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn_output_weights = torch.softmax(scores, dim=-1)
        attn_output_weights = self.relu(self.tanh(attn_output_weights))
        attn_output = torch.matmul(attn_output_weights, value).transpose(1, 2).contiguous().view(batch_size, -1,
                                                                                                 self.d_model)
        attn_output_weights = torch.mean(attn_output_weights, dim=1)
        return attn_output, attn_output_weights
