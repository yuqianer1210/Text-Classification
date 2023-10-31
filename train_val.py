import time

import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from utils.draw import draw_accuracy, draw_loss, draw_f1
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report


# margin = 0.6
# theta = lambda t: (torch.sign(t) + 1.) / 2.
#
#
# # 新Loss  防止过拟合，提高准确率
# def loss(y_true, y_pred):
#     return - (1 - theta(y_true - margin) * theta(y_pred - margin)
#               - theta(1 - margin - y_true) * theta(1 - margin - y_pred)
#               ) * (y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8))


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=torch.tensor([1 / 8000, 1 / 4000])):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        # alpha = self.alpha.cuda()
        pred, target = pred.cpu(), target.cpu()
        log_pt = nn.functional.log_softmax(pred, dim=1)  # 计算softmax后在计算log
        pt = torch.exp(log_pt)  # 对log_softmax去exp，把log取消就是概率
        alpha = self.alpha[target].unsqueeze(dim=1)  # 去取真实索引类别对应的alpha
        log_pt = alpha * (1 - pt) ** self.gamma * log_pt  # focal loss计算公式
        focal_loss = nn.functional.nll_loss(log_pt, target, reduction='sum')  # 最后选择对应位置的元素
        return focal_loss


def train_model(model, config, data_train, data_valid):
    model.train()
    # 先get训练model
    # --- 而后在构建optimizer之前把model导入到gpu之中
    # --- 最后才是构建optimizer，目的是将导入到GPU上的模型参数与优化器进行绑定。
    model.cuda()
    criterion = FocalLoss().cuda()
    # 参数 epsilon = 1e-8 是一个非常小的值，他可以避免实现过程中的分母为 0 的情况
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=1e-8, weight_decay=1e-4)
    total_steps = len(data_train.dataset) * config.epochs  # 总的训练样本数
    warm_up_ratio = 0.1  # 定义要预热的step
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)
    # 记录 模型保存时 的 f1值，epoch值
    f1_best = 0
    epoch_best = 0
    # 记录 训练集 误差、正确率 ， 验证集 误差、正确率
    # 用于 作图
    draw_train_losses = []
    draw_train_acc = []
    draw_val_losses = []
    draw_val_acc = []
    draw_train_f1 = []
    draw_val_f1 = []

    for e in range(config.epochs):
        # labels_all = []
        # predict_all = []
        labels_all = np.array([], dtype=int)
        predict_all = np.array([], dtype=int)
        train_loss = 0  # 重置每次 epoch 的训练总 loss
        # batch loop
        # for inputs, labels, masks in data_train:
        for inputs, labels, masks in tqdm(data_train):
            inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
            model.zero_grad()  # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            # 通过 前向传播 计算 模型的输出 。
            output = model(inputs, masks)  # output: (batch_size, output_size)
            # 计算模型输出与标签之间的交叉熵损失。
            # output.squeeze() 用于去除输出的不必要的维度，labels.long() 用于确保标签是整数类型。
            loss = criterion(output.squeeze(), labels.long())  # 26%
            train_loss += loss.item()  # 累加 loss
            # labels_all.append(labels)
            # predict_all.append(torch.max(output, dim=1)[1])
            labels = labels.cpu().numpy()
            pred = torch.max(output, dim=1)[1].cpu().numpy()  # 获取 预测类别
            labels_all = np.append(labels, labels)
            predict_all = np.append(pred, pred)
            # 模型更新
            loss.backward()  # 反向传播，计算梯度
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，避免出现梯度爆炸情况(太花时间 53%)
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率
            # torch.cuda.empty_cache()  # 清除GPU缓存
        # 计算指标
        P = precision_score(labels_all, predict_all, average="binary")
        R = recall_score(labels_all, predict_all, average="binary")
        train_acc = accuracy_score(labels_all, predict_all)
        F1 = f1_score(labels_all, predict_all, average='binary')
        # target_names = ['Negative', 'Positive']
        # print(classification_report(labels_all, predict_all, target_names=target_names))
        draw_train_losses.append(train_loss)
        draw_train_acc.append(train_acc)
        draw_train_f1.append(F1)
        # ------ 打印训练信息 ------
        print("Epoch: {}/{}, ".format(e + 1, config.epochs),
              "| train_acc: {:.6f} ".format(train_acc),
              "| train_loss: {:.6f} ".format(train_loss),
              "| train_P: {:.6f} ".format(P),
              "| train_R: {:.6f} ".format(R),
              "| train_F1: {:.6f} ".format(F1)
              )
        # ------ 验证模型 -----------
        # 在每个 epoch 结束后，对训练效果进行验证，并打印训练信息
        model.eval()
        val_P, val_R, val_F1, val_acc, val_loss = evaluate(model, config, data_valid)
        model.train()
        draw_val_losses.append(val_loss)  # 添加到 损失列表， 用于作图
        draw_val_acc.append(val_acc)  # 添加到 正确率列表， 用于作图
        draw_val_f1.append(val_F1)
        # ------ 打印验证信息 ------
        print("Epoch: {}/{}, ".format(e + 1, config.epochs),
              "| val_acc: {:.6f} ".format(val_acc),
              "| val_loss: {:.6f} ".format(val_loss),
              "| val_P: {:.6f} ".format(val_P),
              "| val_R: {:.6f} ".format(val_R),
              "| val_F1: {:.6f} ".format(val_F1)
              )
        # 模型保存
        if val_F1 > f1_best:
            f1_best = val_F1  # 记录 f1_score
            epoch_best = e + 1  # 记录 epoch
            print("save model...")
            torch.save(model.state_dict(), config.save_path)  # 保存 模型参数state_dict:训练过程中需要学习的权重和偏执系数
    print("save model at epoch: {}, ".format(epoch_best),
          "val_f1_score:{:.6f}".format(f1_best))
    # 画图
    draw_loss(config, draw_train_losses, draw_val_losses)
    # draw_accuracy(config, draw_train_acc, draw_val_acc)
    draw_f1(config, draw_train_f1, draw_val_f1)


def test_model(model, config, data_test):
    model.load_state_dict(torch.load(config.save_path))
    model.cuda()
    model.eval()
    P, R, F1, Acc, loss = evaluate(model, config, data_test)
    print("Test_Accuracy: {:.6f} ".format(Acc),
          "| Test_Loss: {:.6f} ".format(loss),
          "| Test_P: {:.6f} ".format(P),
          "| Test_R: {:.6f} ".format(R),
          "| Test_F1: {:.6f} ".format(F1)
          )


def evaluate(model, config, data_eval):
    criterion = FocalLoss().cuda()
    loss_total = 0
    labels_all = np.array([], dtype=int)
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for inputs, labels, masks in data_eval:
            inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
            output = model(inputs, masks)
            loss = criterion(output.squeeze(), labels.long())
            loss_total += loss.item()
            labels = labels.cpu().numpy()
            pred = torch.max(output, dim=1)[1].cpu().numpy()  # 获取 预测类别
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, pred)
            # torch.cuda.empty_cache()  # 清除GPU缓存
    # 计算指标
    P = precision_score(labels_all, predict_all, average="binary")
    R = recall_score(labels_all, predict_all, average="binary")
    Acc = accuracy_score(labels_all, predict_all)
    F1 = f1_score(labels_all, predict_all, average='binary')
    # target_names = ['Negative', 'Positive']
    # print(classification_report(labels_all, predict_all, target_names=target_names))
    return P, R, F1, Acc, loss_total
