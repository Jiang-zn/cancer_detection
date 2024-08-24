import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import os
import pandas as pd

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomAffine(degrees=15, scale=(0.8, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

testset = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Data/BreaKHis400X/test', transform=test_transform)
trainloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

print(len(trainloader))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 初始化网络
model = VGG19(num_classes=2, init_weights=False)



model = nn.DataParallel(model)
model.to(device)

model.load_state_dict(torch.load('/content/drive/MyDrive/Data/VGG19_cancer_hc.pth'), strict=False)