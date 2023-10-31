import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))  # w
        self.b_omega = nn.Parameter(torch.Tensor(hidden_dim))  # 偏置项 b
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))  # 查询 Q
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, encoder_outputs):
        u = torch.tanh(torch.matmul(encoder_outputs, self.w_omega) + self.b_omega)  # 键 K
        # (B, L, H) . (H, H) -> (B, L, H)
        att = torch.matmul(u, self.u_omega)  # K*Q
        # (B, L, H) . (H, 1) -> (B, L, 1)
        att_weight = F.softmax(att, dim=1)
        # (B, L, 1) -> (B, L, 1)
        scored_words = encoder_outputs * att_weight
        # (B, L, H) * (B, L, 1) -> (B, L, H)
        context = torch.sum(scored_words, dim=1)
        # (B, L, H) -> (B, H)
        return context, att_weight.squeeze(-1)


# 自注意力
class SelfAttention(nn.Module):
    def __init__(self, attention_size):
        super().__init__()
        self.attention_size = attention_size
        # Self-Attention
        self.key_layer = torch.nn.Linear(self.attention_size, self.attention_size)
        self.query_layer = torch.nn.Linear(self.attention_size, self.attention_size)
        self.value_layer = torch.nn.Linear(self.attention_size, self.attention_size)
        self._norm_fact = 1 / math.sqrt(self.attention_size)

    def forward(self, attention_input):
        K = self.key_layer(attention_input)
        Q = self.query_layer(attention_input)
        V = self.value_layer(attention_input)
        att_weight = torch.nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(att_weight, V)
        output = torch.mean(attention_output, dim=1)

        return output
