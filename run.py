import torch
import numpy as np
import pandas as pd
from importlib import import_module

from DataManager import DataManager
from train_val import train_model, test_model
from predict import predict


import random
from line_profiler import LineProfiler


# 设置随机种子，保证结果每次结果一样
# 固定cuda的随机数种子，每次返回的卷积算法将是确定的
torch.backends.cudnn.deterministic = True
# 提高运行效率
# torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(0)


if __name__ == '__main__':

    # 选择模型
    model_name = 'bert'  # 0.829710
    # model_name = 'bert_bilstm'  # 0.943557
    # model_name = 'bert_textcnn'  #
    # model_name = 'bert_bilstm_textcnn'  #

    # 动态加载模块
    x = import_module('models.' + model_name)
    # 加载 配置参数
    config = x.ModelConfig()
    #
    config.use_cuda = USE_CUDA
    # 指定 数据集
    if config.dataset:
        # 9583条 数据        len=200
        data = pd.read_csv('dataset/ChnSentiCorp/train.tsv', sep='\t', encoding='utf-8')
    else:
        # 1w条 数据    positive: 5000   negative: 5000    len=192
        data = pd.read_csv('dataset/NLPCC14-SC/train.tsv', sep='\t', encoding='utf-8')

    # 数据处理
    print('read data...')
    dm = DataManager(data, config)
    # dm.count_text_len()  # 得到 数据集 句子长度图  200/192
    train_loader, valid_loader, test_loader = dm.get_data()

    # 模型加载
    # print("load model...")
    # model = x.Model(config)

    # 判断 是否 使用GPU 训练
    # if config.use_cuda:
    #     print('Run on GPU.')
    # else:
    #     print('No GPU available, run on CPU.')

    # 训练
    # print("train model...")
    # train_model(model, config, train_loader, valid_loader)

    # 逐句监控 运行时间
    # numbers = [random.randint(1, 100) for i in range(1000)]
    # lp = LineProfiler()
    # lp_wrapper = lp(train_model)
    # lp_wrapper(model, config, train_loader, valid_loader)
    # lp.print_stats()
    # lp.dump_stats('saveName.lprof')

    # 测试
    # print("test model...")
    # test_model(model, config, test_loader)

    # 预测
    # print("predict...")
    # comments = ['这个菜不太行']
    # print(comments)
    # inputs_ids, attention_mask = dm.get_data_predict(comments)
    # predict(model, config, inputs_ids, attention_mask)
