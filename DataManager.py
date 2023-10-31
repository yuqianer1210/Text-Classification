import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from prefetch_generator import BackgroundGenerator
from transformers import BertTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# 预处理 数据集
# 剔除标点符号,\xa0 空格
def pretreatment(comments):
    result_comments = []
    punctuation = '。，？！@#￥*：%&~（）、；“”&|,.?!:%&~();""'
    for comment in comments:
        comment = ''.join([c for c in comment if c not in punctuation])  # 去标点符号
        comment = ''.join(comment.split())  # 去 \xa0 空格
        result_comments.append(comment)

    return result_comments


class DataManager(object):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)

        self.inputs = list(self.data['text_a'].values)
        self.labels = torch.from_numpy(self.data['label'].values).float()

    # 获取 数据集 句子长度
    def count_text_len(self):
        text_len = []
        max_len = 1000
        for comment in self.inputs:
            if len(comment) > max_len:
                print(comment)
            text_len.append(len(comment))
        print(len(text_len))
        print(max(text_len))
        print(np.mean(text_len))
        # self.data['text_len'] = self.data['text_a'].apply(lambda x: len(x.split(' ')))
        # print(self.data['text_len'].describe())
        # self.data['label'].value_counts().plot(kind='box')
        plt.hist(text_len)  # 作图
        # my_x_ticks = np.arange(0, 800, 64)  # 设置坐标轴刻度
        # plt.xticks(my_x_ticks)
        # my_y_ticks = np.arange(0, 10000, 1000)  # 设置坐标轴刻度
        # plt.yticks(my_y_ticks)
        plt.grid(True)  # 网格线
        if self.config.dataset == 1:
            save_path = './picture/ChnSentiCorp.png'
        else:
            save_path = './picture/NLPCC14-SC.png'

        # save_path = './Picture/weibo_sinti_100k.png'
        plt.savefig(save_path)
        plt.show()

    # 数据封装
    def data_loder(self, x, y):
        # tokenized
        x_tokenized = self.tokenizer(x,
                                     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                     padding="max_length",
                                     max_length=self.config.max_len,
                                     truncation=True,
                                     return_tensors='pt')
        x_ids = x_tokenized['input_ids']
        attention_mask = x_tokenized['attention_mask']
        # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
        data = TensorDataset(x_ids, y, attention_mask)
        result = DataLoaderX(data,
                             batch_size=self.config.batch_size,
                             sampler=RandomSampler(data),  # 随机小批量
                             num_workers=8,
                             pin_memory=True,
                             drop_last=True)
        return result

    # 获取 训练集、验证集、测试集 数据
    def get_data(self):
        # 划分 数据集  ————  训练集：验证集：测试集 == 8:1:1
        X_train, X_test, y_train, y_test = train_test_split(self.inputs,
                                                            self.labels,
                                                            test_size=0.2,
                                                            shuffle=True,
                                                            stratify=self.labels,
                                                            random_state=0)
        X_valid, X_test, y_valid, y_test = train_test_split(X_test,
                                                            y_test,
                                                            test_size=0.5,
                                                            shuffle=True,
                                                            stratify=y_test,
                                                            random_state=0)
        # 数据封装
        train_loader = self.data_loder(X_train, y_train)
        valid_loader = self.data_loder(X_valid, y_valid)
        test_loader = self.data_loder(X_test, y_test)

        return train_loader, valid_loader, test_loader

    # 获取 处理后的 预测数据
    def get_data_predict(self, comments_list):
        # 预处理去掉标点符号
        result_comments = pretreatment(comments_list)
        # tokenized
        results = self.tokenized(result_comments)

        input_ids = results[0]
        attention_mask = results[1]

        return input_ids, attention_mask

    # 显示参数 stratify 作用
    def show_stratify(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        # un-stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            self.inputs, self.labels, test_size=0.2, random_state=0, stratify=None)
        plot_class_distribution(y_train, y_test, ax1, 'Not stratified (`stratify=None`)')
        # stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            self.inputs, self.labels, test_size=0.2, random_state=0, stratify=labels)
        plot_class_distribution(y_train, y_test, ax2, 'Stratified (`stratify=y`)')
        plt.show()


# 显示 数据 类别分布
def plot_class_distribution(y_train, y_test, ax, title=None):
    y_train, y_test = y_train.numpy(), y_test.numpy()
    count = [[(y_train == label).sum() / len(y_train), label, 'Train'] for label in np.unique(y_train)]
    count += [[(y_test == label).sum() / len(y_test), label, 'Test'] for label in np.unique(y_test)]
    count = pd.DataFrame(count, columns=['Ratio', 'Class Label', 'Split'])
    count = count.sort_values(by='Ratio', ascending=False)
    sns.barplot(data=count, x='Class Label', y='Ratio', hue='Split', ax=ax)
    ax.set_title(title)
    return ax
