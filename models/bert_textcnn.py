import torch
import torch.nn as nn
from transformers import BertModel


class ModelConfig:
    """配置参数"""
    # 训练 参数
    model_name = 'bert_textcnn'  # 模型
    dataset = 1  # 数据集    0： 2000条数据   1： 1w+ 数据

    epochs = 6  # 2-4 设置过大容易过拟合
    # 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
    batch_size = 32  # 批量大小
    lr = 5e-5  # 5e-5, 3e-5, 2e-5

    max_len = 88  # 填充长度
    clip = 5  # gradient clipping
    bert_path = 'bert-base-chinese'  # 预训练bert路径
    save_path = r'E:\PyCharm\TextClassification_SentimentAnalysis\Parameters\bert_textcnn.pth'  # 模型保存路径
    use_cuda = False  # 默认 不使用GPU

    # Bert
    hidden_size = 768  # bert 标准特征维度
    # TextCNN
    num_classes = 2  # 分类 类别个数
    filter_sizes = (3, 4, 5)  # 卷积核尺寸
    num_filters = 128  # 卷积核数量（channels数）
    dropout = 0.5  # 丢弃率


# bert+textcnn
class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # bert
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # TextCNN
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_classes
        self.num_filter_total = config.num_filters * len(config.filter_sizes)
        self.filter_list = torch.nn.ModuleList([
            torch.nn.Conv2d(1, config.num_filters,
                            kernel_size=(size, config.hidden_size)) for size in config.filter_sizes
        ])
        # FNN
        self.dropout = nn.Dropout(config.dropout)
        self.block = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(self.num_filter_total, 128),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(128, config.num_classes),
            # nn.Dropout(config.dropout),
            # nn.Softmax(dim=1)
        )

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
        # self.bert(input, mask)[0] 等价于 self.bert(input, mask).last_hidden_state
        last_hidden_state = self.bert(input_ids, attention_mask).last_hidden_state
        x = self.dropout(last_hidden_state)
        # TextCNN——Classify
        # x: [batch_size, 12, hidden]
        # 添加1个维度
        x = x.unsqueeze(1)  # [batch_size, channel=1, 12, hidden]
        # 对3个尺寸卷积核得到的 feature_map 在第1个维度上 进行拼接
        # 输入： [batch_size, num_filters=256]*len(filter_sizes)=3
        # 输出： [batch_size, num_filter_total]
        # num_filter_total=num_filters*len(filter_sizes)=256*3
        out = torch.cat([self.conv_and_pool(x, conv) for conv in self.filter_list], 1)
        out = self.block(out)  # [batch_size, class_num]

        return out
