import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision.datasets as dset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
import time
import os


# 3×3卷积
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=dilation,
                     dilation=dilation, groups=groups)


# 1×1卷积，起到降维或者升维作用，通常在resnet50、resnet101中使用
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 普通Block模块
# 主要在resnet18、34中使用
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64!")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock!")
        # 当stride步长不为0时，self.conv1和self.downsample会对输入x进行下采样
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 实际上是在进行欠采样，在网络的某些层可能会用到
        if self.downsample is not None:
            identity = self.downsample(x)
        # 在输出上叠加了输入x
        out = out + identity
        out = self.relu(out)

        return out


# Bottleneck模块
# 主要在resnet50、101中使用
# 主要目的是为了减少参数的数量，从而减少计算量，且在降维之后可以更加有效、直观地进行数据的训练和特征提取
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:  # 如果没有传入BatchNorm层，直接设置BatchNorm
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 定义整个残差网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=1000, zero_init_residual=False, init_weight=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 元组里每个元素代表是否将对应的 2x2 stride更换为空洞卷积
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # 第一部分卷积模块
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True);
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.FC = nn.Linear(512 * block.expansion, num_class)
        # 初始化
        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # 对每个残差模块最后一个BN层进行0初始化
        # 使得每个残差模块在最开始的时候类似于恒等连接，从而进一步提升性能
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    # 生成多个卷积层，形成一个大的模块
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.FC(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, num_classes, **kwargs):
    model = ResNet(block, layers, num_class=num_classes, **kwargs)
    return model


# ResNet34
def resnet34(num_classes=1000, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)


# ResNet50
def resnet50(num_classes=1000, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


# ResNet101
def resnet101(num_classes=1000, **kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)


if __name__ == '__main__':
    torch.manual_seed(1)
    Model = resnet50(2)

    # # 使用部分预训练VGG模型 方法2
    # Model.load_state_dict(torch.load('pretrained/resnet101-5d3b4d8f.pth'), strict=False)

    # for name, para in Model.named_parameters():
    #     print(name, torch.min(para))


