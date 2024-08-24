from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision.datasets as dset
import torchvision
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms, models, datasets
import time
import os
from torchvision import datasets, models, transforms


# VGG19网络
# vgg16、19，在3、4、5block块中各减少一个卷积层
class VGG16(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        # 因为前面可以用预训练模型参数，所以单独把最后一层提取出来
        self.classifier2 = nn.Linear(4096, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # torch.flatten 推平操作
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.classifier2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


model = VGG16(num_classes=2, init_weights=True)

# 训练模型


# 串联多个图片变换的操作,调节尺寸
train_transform = transforms.Compose([
    # 将图像短边缩放至256
    # 将图像随机裁剪为统一大小224
    # 对图像进行随机的仿射变换
    # 将一个PIL Image格式的图片转换为tensor格式的图片
    # 对图片进行标准化
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomAffine(degrees=15, scale=(0.8, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    # 将图像短边缩放至256
    # 从中心裁剪出224的图像
    # 将一个PIL Image格式的图片转换为tensor格式的图片
    # 对图片进行标准化
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# shuffle，默认为false，在每次迭代训练时将数据打乱的话则改为true
trainset = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Data/new_BreaKHis/train',
                                            transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valset = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Data/new_BreaKHis/val', transform=val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False)
print(len(trainloader), len(valloader))

# 训练模型

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 初始化网络
model = VGG16(num_classes=2, init_weights=False)

# 保存模型中的weight权值和bias偏置值
# 加载模型参数
model_dict = model.state_dict()
model.load_state_dict(torch.load('/content/drive/MyDrive/Data/vgg16-397923af.pth'), strict=False)
model = nn.DataParallel(model)
model.to(device)

Lr = 0.001
Momentum = 0.99
# 定义loss function和优化器，可对优化器进行调参，或选择Adam优化器，损失函数也可不止局限于交叉熵损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=Lr, momentum=Momentum)

# 保存每个epoch后的Accuracy Loss
Accuracy = []
Loss = []
Val_Accuracy = []
BEST_VAL_ACC = 0.
# 训练
since = time.time()
# epoch也可迭代不止10次，可调
for epoch in range(20):
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
        '/content/drive/MyDrive/Data/image/vgg16/{:.4f}_{:.4f}_Train_accuracy_vs_epochs.png'.format(Lr, Momentum))
    plt.clf()
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(epoch + 1), Loss)
    ax2.set_title("Average trainset loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")
    plt.savefig('/content/drive/MyDrive/Data/image/vgg16/{:.4f}_{:.4f}_loss_vs_epochs.png'.format(Lr, Momentum))

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
        torch.save(model.state_dict(), '/content/drive/MyDrive/Data/VGG19_cancer.pth')
        BEST_VAL_ACC = acc
        print('Saved!')

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch + 1), Val_Accuracy)
    ax3.set_title("Average Val accuracy vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current Val accuracy")
    plt.savefig('/content/drive/MyDrive/Data/image/vgg16/{:.4f}_{:.4f}_val_accuracy_vs_epoch.png'.format(Lr, Momentum))
    plt.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Now the best val Acc is {:.4f}%'.format(BEST_VAL_ACC))
print('当前SGD优化函数参数学习率为{:.4f}，冲量为{:.4f}'.format(Lr, Momentum))
