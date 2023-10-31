import torch.nn as nn
from transformers import BertModel


class ModelConfig:
    """配置参数"""
    # 训练参数
    model_name = 'bert'  # 模型名称
    dataset = 0  # 选取数据集     1： 1w+ 数据   0： 2000条数据

    epochs = 30  # 2-4 设置过大容易过拟合   5
    batch_size = 16  # 批量大小
    lr = 2e-5  # 5e-5, 3e-5, 2e-5

    print_every = 10
    clip = 5  # gradient clipping
    use_cuda = False  # 默认 不使用GPU
    bert_path = 'bert-base-chinese'  # 预训练bert路径
    save_path = r'E:\PyCharm\TextClassification_SentimentAnalysis\Parameters\bert.pth'  # 模型保存路径
    # Bert
    max_len = 192  # 填充长度
    hidden_size = 768
    # FC
    dropout = 0.5
    num_classes = 2  # 分类 个数


# bert + fnn
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # FNN
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        # inference
        output_bert = self.bert(input_ids, attention_mask)  # (batch_size, sen_length, hidden_size)
        # 取出 pooler_output ，(pooler_output 将 [CLS]这个token经过 全连接+tanh ，作为句子的特征向量)
        output_pooler = output_bert.pooler_output
        # fc
        out_drop = self.dropout(output_pooler)
        out = self.fc(out_drop)

        return out
