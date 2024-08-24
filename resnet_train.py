import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import os
import pandas as pd

from cancer_detection.prepare_resnet import *

# 训练ResNet模型
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomAffine(degrees=15, scale=(0.8, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainset = torchvision.datasets.ImageFolder(root='D:/python-Project/cancer_detection/new_BreaKHis/train',
                                            transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
valset = torchvision.datasets.ImageFolder(root='D:/python-Project/cancer_detection/new_BreaKHis/val',
                                          transform=val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
print(len(trainloader), len(valloader))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 初始化网络
model = resnet50(2)
# 下载预训练模型
model.load_state_dict(torch.load('D:/python-Project/cancer_detection/resnet50-19c8e357.pth'), strict=False)
model = nn.DataParallel(model)
model.to(device)
Lr = 0.001
Momentum = 0.95
# 定义loss function和优化器，可对优化器进行调参，或选择Adam优化器，损失函数也可不止局限于交叉熵损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=Lr, momentum=Momentum)
print('当前SGD优化函数参数学习率为{:.4f}，冲量为{:.4f}'.format(Lr, Momentum))
# 保存每个epoch后的Accuracy Loss
Accuracy = []
Loss = []
Val_Accuracy = []
BEST_VAL_ACC = 0.
# 训练
# 打印开始时间，结束时间，总共花费的时间
start = time.time()
print('Start time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# epoch也可迭代不止10次，可调
for epoch in range(20):
    epoch_start = time.time()
    train_loss = 0.
    train_accuracy = 0.
    run_accuracy = 0.
    run_loss = 0.
    total = 0.
    model.train()
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # 经典四步
        # 把梯度置零，也就是把loss关于weight的导数变成0
        # 前向传播求出预测的值
        # 求loss
        # 反向传播求梯度
        # 更新所有参数
        optimizer.zero_grad()
        outs = model(images)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        # 输出状态
        total += labels.size(0)
        run_loss += loss.item()
        _, prediction = torch.max(outs, 1)
        run_accuracy += (prediction == labels).sum().item()
        if i % 20 == 19:
            print('epoch {},iter {},train accuracy: {:.4f}%   loss:  {:.4f}'.format(epoch, i + 1, 100 * run_accuracy / (
                    labels.size(0) * 20), run_loss / 20))
            train_accuracy += run_accuracy
            train_loss += run_loss
            run_accuracy, run_loss = 0., 0.
    Loss.append(train_loss / total)
    Accuracy.append(100 * train_accuracy / total)
    # 可视化训练结果
    fig1, ax1 = plt.subplots(figsize=(11, 8))
    ax1.plot(range(0, epoch + 1, 1), Accuracy)
    ax1.set_title("Average trainset accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. train. accuracy")
    plt.savefig(
        'D:/python-Project/cancer_detection/image/{:.4f}_{:.4f}_Train_accuracy_vs_epochs.png'.format(Lr,
                                                                                                     Momentum))
    plt.clf()
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(epoch + 1), Loss)
    ax2.set_title("Average trainset loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")
    plt.savefig(
        'D:/python-Project/cancer_detection/image/{:.4f}_{:.4f}_loss_vs_epochs.png'.format(Lr, Momentum))

    plt.clf()
    plt.close()

    # 验证
    acc = 0.
    model.eval()
    print('等待验证...')
    with torch.no_grad():
        accuracy = 0.
        total = 0
        for data in valloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            _, prediction = torch.max(out, 1)
            total += labels.size(0)
            accuracy += (prediction == labels).sum().item()
            acc = 100. * accuracy / total
    print('epoch {}  The ValSet accuracy is {:.4f}% \n'.format(epoch, acc))
    Val_Accuracy.append(acc)
    if acc > BEST_VAL_ACC:
        print('Find Better Model and Saving it...')
        torch.save(model.state_dict(), 'D:/python-Project/cancer_detection/new_BreaKHis/ResNet50_cancer.pth')
        BEST_VAL_ACC = acc
        print('Saved!')

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch + 1), Val_Accuracy)
    ax3.set_title("Average Val accuracy vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current Val accuracy")
    plt.savefig(
        'D:/python-Project/cancer_detection/image/{:.4f}_{:.4f}_val_accuracy_vs_epoch.png'.format(Lr,
                                                                                                  Momentum))
    plt.close()
    print('Now the best val Acc is {:.4f}%'.format(BEST_VAL_ACC))
    epoch_end = time.time()
    print('Training complete in : ', int((epoch_end - epoch_start) / 3600), 'h',
          int((epoch_end - epoch_start) % 3600 / 60), 'm', int((epoch_end - epoch_start) % 60),
          's')

print('End time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
end = time.time()
# 计算耗时，小时，分钟，秒
print('Total time: ', int((end - start) / 3600), 'h', int((end - start) % 3600 / 60), 'm', int((end - start) % 60), 's')
