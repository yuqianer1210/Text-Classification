import torch
import numpy as np
from tqdm import tqdm

from utils.draw import draw_accuracy, draw_loss, draw_f1


def train_model(model, config, data_train, data_valid):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.use_cuda:
        model.cuda()
    model.train()
    # 记录 训练集 误差、正确率 ， 验证集 误差、正确率
    # 用于 作图
    draw_train_losses = []
    draw_train_acc = []
    draw_val_losses = []
    draw_val_acc = []

    for e in range(config.epochs):
        train_loss = 0  # 重置每次 epoch 的训练总 loss
        num_train_acc = 0  # 重置 标签预测正确 数
        # batch loop
        for inputs, labels, masks in tqdm(data_train):
            if config.use_cuda:
                inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
            # 通过 前向传播 计算 模型的输出 。
            output = model(inputs, masks)  # output: (batch_size, output_size)
            torch.cuda.empty_cache()  # 清除GPU缓存
            # 计算模型输出与标签之间的交叉熵损失。
            # output.squeeze() 用于去除输出的不必要的维度，labels.long() 用于确保标签是整数类型。
            loss = criterion(output.squeeze(), labels.long())
            train_loss += loss.item()  # 累加 loss
            num_acc = (output.argmax(dim=1) == labels).sum().item()  # 计算 正确分类个数
            num_train_acc += num_acc
            # 模型更新
            model.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播，计算梯度
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，避免出现梯度爆炸情况
            optimizer.step()  # 更新参数
        # ------ 验证模型 -----------
        # 在每个 epoch 结束后，对训练效果进行验证，并打印训练信息
        model.eval()
        # 重置 每个epoch 验证集误差 和 正确个数
        total_val_loss = 0
        num_val_acc = 0
        # 在此上下文中不计算梯度。
        with torch.no_grad():
            for inputs, labels, masks in data_valid:
                if config.use_cuda:
                    inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
                output = model(inputs, masks)  # 前向传播，计算模型在验证集上的输出
                torch.cuda.empty_cache()  # 清除GPU缓存
                val_loss = criterion(output.squeeze(), labels.long())  # 计算验证集上的损失
                total_val_loss += val_loss.item()  # 累加 loss
                num_acc = (output.argmax(dim=1) == labels).sum().item()  # 正确个数
                num_val_acc += num_acc
        draw_val_losses.append(total_val_loss)  # 添加到 损失列表， 用于作图
        val_acc = num_val_acc / len(data_valid.dataset)  # 每个epoch，验证集正确率
        draw_val_acc.append(val_acc)  # 添加到 正确率列表， 用于作图

        draw_train_losses.append(train_loss)
        train_acc = num_train_acc / len(data_train.dataset)
        draw_train_acc.append(train_acc)
        # 打印训练和验证信息：
        print("Epoch: {}/{}, ".format(e + 1, config.epochs),
              "| Training Loss: {:.6f}, ".format(train_loss),
              "| Training Accuracy: {:.6f}, ".format(train_acc),
              "| Val Loss: {:.6f}".format(total_val_loss),
              "| Val Accuracy: {:.6f}".format(val_acc)
              )
        model.train()

    # 只保存 模型的参数
    # state_dict变量 ：存放训练过程中需要学习的权重和偏执系数
    torch.save(model.state_dict(), config.save_path)

    draw_loss(config, draw_train_losses, draw_val_losses)
    draw_accuracy(config, draw_train_acc, draw_val_acc)


def test_model(model, config, data_test):
    model.load_state_dict(torch.load(config.save_path))
    if config.use_cuda:
        model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    test_losses = 0  # track loss
    num_correct = 0
    model.eval()
    # 在此上下文中不计算梯度。
    with torch.no_grad():
        # iterate over test data
        for inputs, labels, masks in data_test:
            # h = tuple([each.data for each in h])
            if config.use_cuda:
                inputs, labels, masks = inputs.cuda(), labels.cuda(), masks.cuda()
            output = model(inputs, masks)
            torch.cuda.empty_cache()  # 清除GPU缓存
            test_loss = criterion(output.squeeze(), labels.long())
            test_losses += test_loss.item()
            pred = torch.max(output, dim=1)[1]  # 获取 预测类别
            # compare predictions to true label
            correct_tensor = pred.eq(labels.long().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not config.use_cuda else np.squeeze(
                correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)
        print("Test loss: {:.3f}".format(test_losses))
        # accuracy over all test data
        test_acc = num_correct / len(data_test.dataset)
        print("Test accuracy: {:.3f}".format(test_acc))
