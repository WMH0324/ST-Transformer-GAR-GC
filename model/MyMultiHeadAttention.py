# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class MyMultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super(MyMultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.d_model = d_model
#         self.dropout = dropout
#
#         self.query_linear = nn.Linear(d_model, d_model)
#         self.key_linear = nn.Linear(d_model, d_model)
#         self.value_linear = nn.Linear(d_model, d_model)
#         self.output_linear = nn.Linear(d_model, d_model)
#
#     def forward(self, query, key, value):
#         batch_size = query.size(0)
#
#         # Linear transformations
#         query = self.query_linear(query)
#         key = self.key_linear(key)
#         value = self.value_linear(value)
#
#         # Reshape for multi-head attention
#         query = query.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
#         key = key.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
#         value = value.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
#
#         # Compute attention scores
#         scores = torch.matmul(query, key.transpose(-2, -1))
#         scores = scores / torch.sqrt(torch.tensor(self.d_model // self.num_heads, dtype=torch.float32))
#         attention_weights = F.dropout(torch.softmax(scores, dim=-1), p=self.dropout, training=self.training)
#
#         # Apply attention weights to value
#         weighted_values = attention_weights.matmul(value)
#
#         # Reshape and concatenate heads
#         weighted_values = weighted_values.view(batch_size, -1, self.d_model)
#
#         # Linear transformation for output
#         output = self.output_linear(weighted_values)
#
#         return output
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MyMultiheadAttention, self).__init__()
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
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn_output_weights = torch.softmax(scores, dim=-1)


        attn_output = torch.matmul(attn_output_weights, value).transpose(1, 2).contiguous().view(batch_size, -1,
                                                                               self.d_model)
        attn_output_weights = torch.mean(attn_output_weights, dim=1)
        return attn_output, attn_output_weights
