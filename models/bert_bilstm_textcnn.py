import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
from models.attention import SelfAttention


class ModelConfig:
    """配置参数"""
    # 训练 参数
    model_name = 'bert_bilstm_textcnn'  # 模型
    dataset = 0  # 数据集    0： 1w条数据   1： 9600 数据

    epochs = 4  # 2-4 设置过大容易过拟合
    batch_size = 16  # 批量大小
    lr = 5e-5  # 5e-5, 3e-5, 2e-5

    max_len = 192  # 填充长度
    clip = 5  # gradient clipping
    use_cuda = False  # 默认 不使用GPU
    bert_path = 'bert-base-chinese'  # 预训练bert路径
    save_path = r'E:\PyCharm\TextClassification_SentimentAnalysis\Parameters\bert_bilstm_textcnn.pth'  # 模型保存路径

    # Bert
    hidden_size = 768  # bert 标准特征维度
    # TextCNN
    num_classes = 2  # 分类 类别个数
    filter_sizes = (3, 4, 5)  # 卷积核尺寸
    num_filters = 128  # 卷积核数量（channels数）
    # BiLSTM
    n_layers = 8
    output_size = 128
    # FNN
    dropout = 0.5  # 丢弃率


# Bert+TextCNN+BiLSTM
class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.num_filter_total = config.num_filters * len(config.filter_sizes)
        self.attention_size = self.num_filter_total + config.output_size * 2
        # bert
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # TextCNN
        self.filter_list = torch.nn.ModuleList([
            torch.nn.Conv2d(1, config.num_filters,
                            kernel_size=(size, config.hidden_size)) for size in config.filter_sizes
        ])
        # LSTM
        self.lstm = torch.nn.LSTM(input_size=config.hidden_size, hidden_size=config.output_size,
                                  num_layers=config.n_layers,
                                  bias=True, batch_first=True, bidirectional=True)
        # 设置lstm隐层状态h0, c0,作为模型参数的一部分进行优化
        state_size = (config.n_layers * 2, config.batch_size, config.output_size)
        self.init_h = nn.Parameter(torch.zeros(state_size))
        self.init_c = nn.Parameter(torch.zeros(state_size))
        # Self-Attention
        self.self_attention = SelfAttention(self.attention_size)
        # FNN
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.attention_size, config.num_classes)

    def conv_and_pool(self, x, conv):
        # 卷积
        x = torch.nn.functional.relu(conv(x)).squeeze(3)  # [batch_size, num_filters=256, 12-filter_sizes[i]+1]
        # 1-MAX 池化 (对第2个维度进行池化)
        x = torch.nn.functional.max_pool1d(x, x.size(2)).squeeze(2)  # [batch_size, num_filters=256]
        # [batch_size, num_filters=256]
        return x

    def forward(self, input_ids, attention_mask):
        # Bert——Embedding
        # 取出  last_hidden_state     [batch_size, seq_len, hidden]
        last_hidden_state = self.bert(input_ids, attention_mask).last_hidden_state
        # TextCNN
        # x: [batch_size, 12, hidden]
        x = last_hidden_state.unsqueeze(1)  # 添加一个维度 [batch_size, channel=1, 12, hidden]
        # 对3个尺寸卷积核得到的 feature_map 在第1个维度上 进行拼接
        # 输入： [batch_size, num_filters]*len(filter_sizes)
        # 输出： [batch_size, num_filter_total]
        textcnn_output = torch.cat([self.conv_and_pool(x, conv) for conv in self.filter_list], 1)
        # BiLSTM
        lstm_out, (hidden_last, cn_last) = self.lstm(last_hidden_state, (self.init_h, self.init_c))
        # 正向最后一层，最后一个时刻
        hidden_last_L = hidden_last[-2]  # print(hidden_last_L.shape)  #[batch_size, 192]
        # 反向最后一层，最后一个时刻
        hidden_last_R = hidden_last[-1]  # print(hidden_last_R.shape)   #[batch_size, 192]
        # 拼接
        bilstm_output = torch.cat([hidden_last_L, hidden_last_R], dim=-1)  # bilstm_output ：[batch_size, 384]
        # 拼接 TextCNN BiLSTM
        out_concat = torch.cat([textcnn_output, bilstm_output], dim=1)  # [batch_size, attention_size(768)]
        # self-attention
        att_input = out_concat.unsqueeze(1)  # [batch_size, 1, attention_size]
        att_output = self.self_attention(att_input)
        # fnn
        out = self.dropout(att_output)
        out = self.fc(out)

        return out
