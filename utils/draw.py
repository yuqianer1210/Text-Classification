import matplotlib.pyplot as plt
import numpy as np


def draw_loss(config, train_loss, val_loss):
    epochs = config.epochs
    x = range(1, epochs + 1)
    # 创建窗口
    plt.figure()
    # train_loss
    plt.plot(x, train_loss, 'r', label='train loss')
    # val_loss
    plt.plot(x, val_loss, 'g', label='val loss')
    plt.xlim((0, epochs + 1))  # 设置坐标轴范围
    my_x_ticks = np.arange(1, epochs + 1, 1)  # 设置坐标轴刻度
    plt.xticks(my_x_ticks)
    plt.grid(True)  # 网格线
    plt.title('Train loss and Validation loss vs. epoches', fontsize=20)
    plt.legend()  # 图例
    # 保存
    save_path = './picture/' + config.model_name + '_loss.png'
    plt.savefig(save_path)


def draw_accuracy(config, train_acc, val_acc):
    epochs = config.epochs
    x = range(1, epochs + 1)
    plt.title('Train acc and Validation acc vs. epoches', fontsize=20)
    # 创建窗口
    plt.figure()
    # train_loss
    plt.plot(x, train_acc, 'r', label='train acc')
    # val_loss
    plt.plot(x, val_acc, 'g', label='val acc')
    # 设置坐标轴范围
    plt.xlim((0, epochs + 1))
    # 设置坐标轴刻度
    my_x_ticks = np.arange(1, epochs + 1, 1)
    plt.xticks(my_x_ticks)
    # 网格线
    plt.grid(True)
    # 图例
    plt.legend()
    # 保存
    save_path = './picture/' + config.model_name + '_acc.png'
    plt.savefig(save_path)


def draw_f1(config, train_f1, val_f1):
    epochs = config.epochs
    x = range(1, epochs + 1)
    plt.title('Train f1_score and Validation f1_score vs. epoches', fontsize=20)
    # 创建窗口
    plt.figure()
    # train_loss
    plt.plot(x, train_f1, 'r', label='train f1_score')
    # val_loss
    plt.plot(x, val_f1, 'g', label='val f1_score')
    # 设置坐标轴范围
    plt.xlim((0, epochs + 1))
    # 设置坐标轴刻度
    my_x_ticks = np.arange(1, epochs + 1, 1)
    plt.xticks(my_x_ticks)
    # 网格线
    plt.grid(True)
    # 图例
    plt.legend()
    # 保存
    save_path = './picture/' + config.model_name + '_f1_score.png'
    plt.savefig(save_path)


