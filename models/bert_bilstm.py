import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel
from models.attention import Attention


class ModelConfig:
    """配置参数"""
    # 训练参数
    model_name = 'bert_bilstm'  # 模型名称
    bert_path = 'bert-base-chinese'  # 预训练bert路径
    save_path = r'E:\PyCharm\TextClassification_SentimentAnalysis\Parameters\bert_bilstm.pth'  # 模型保存路径
    use_cuda = False  # 默认 不使用GPU
    clip = 5  # gradient clipping

    dataset = 1  # 选取数据集     1： 1w+ 数据   0： 2000条数据
    max_len = 192  # 填充长度

    batch_size = 16  # 批量大小
    epochs = 20  # 2-4 设置过大容易过拟合   4
    lr = 2e-5  # 5e-5, 3e-5, 2e-5

    # BiLSTM
    hidden_size = 768
    output_size = 128
    # 设置为64时，出现bug：Precision is ill-defined and being set to 0.0 due to no predicted samples.
    # 设置为32时， 训练集 和 验证集 Loss 均不下降
    n_layers = 2
    # FNN
    num_classes = 2  # 分类 类别数
    dropout = 0.5


# def attention_net(x, query, mask=None):  # 缩放点积注意力（k=v） x->[128,52,128]
#     d_k = query.size(-1)  # batch_size,seq_len,embedding_dim
#     # 打分机制 scores:[batch, seq_len, seq_len]
#     scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # [batch,seq_len,seq_len]
#     p_attn = F.softmax(scores, dim=-1)  # 沿着列方向进行softmax [batch,seq_len,seq_len] ->[128,52,52]
#     context = torch.matmul(p_attn, x).sum(1)  # 将注意力矩阵与输入序列相乘再求和 ->[128,52,52]
#     return context, p_attn


# bert+bilstm
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.attention_size = config.output_size * 2
        # bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # BiLSTM
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.output_size,
                            num_layers=config.n_layers, bias=True, batch_first=True, bidirectional=True)
        # 设置lstm隐层状态h0, c0,作为模型参数的一部分进行优化
        state_size = (config.n_layers * 2, config.batch_size, config.output_size)
        self.init_h = nn.Parameter(torch.zeros(state_size))
        self.init_c = nn.Parameter(torch.zeros(state_size))
        # Attention
        self.attention = Attention(self.attention_size)
        # FNN
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.attention_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        # 生成bert字向量 : 取出 last_hidden_state
        last_hidden_state = self.bert(input_ids, attention_mask).last_hidden_state
        # lstm_out : 每一时间步的 最后一层的 输出 h（按位置拼接后的向量）
        lstm_out, _ = self.lstm(last_hidden_state, (self.init_h, self.init_c))
        # print(lstm_out.shape)   #[16,100,768]
        # attention
        att_out, att_weights = self.attention(lstm_out)
        # fnn
        out = self.dropout(att_out)
        # out = F.relu(out)
        out = self.fc(F.relu(out))  # print(out.shape)    #[16,2]

        return out

