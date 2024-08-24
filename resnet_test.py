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

# 测试模型


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
model = resnet50(2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load('D:/python-Project/cancer_detection/new_BreaKHis/ResNet50_cancer.pth'))

# 测试
# id_list_benign = []
# pred_list_benign = []
# id_list_malignant = []
# pred_list_malignant = []
id_list_all = []
pred_list_all = []
# path_benign = '/content/drive/MyDrive/Data/new_BreaKHis/test/new_benign/'
# path_benign_file = os.listdir(path_benign)
# path_malignant = '/content/drive/MyDrive/Data/new_BreaKHis/test/new_malignant/'
# path_malignant_file = os.listdir(path_malignant)

path_mix = '/content/drive/MyDrive/Data/new_BreaKHis/mymix/mix/'
path_mix_file = os.listdir(path_mix)

model.eval()
with torch.no_grad():
    for file in tqdm(path_mix_file):
        img_mix = Image.open(path_mix + file)
        string_id = str(file.split('.')[0])
        img_mix = transform(img_mix)
        # 增加维度
        img_mix = img_mix.unsqueeze(0)
        img_mix = img_mix.to(device)
        out_mix = model(img_mix)
        prediction_mix = F.softmax(out_mix, dim=1)[:, 1].tolist()
        _predict_mix = np.array(prediction_mix)
        # 将所有概率大于0.5的元素替换为1，否则替换为0
        _predict_mix = np.where(_predict_mix > 0.5, 1, 0)
        # print(string_id, _predict_mix[0])
        id_list_all.append(string_id)
        pred_list_all.append(_predict_mix)

res = pd.DataFrame({
    'id': id_list_all,
    'label': pred_list_all
})

res.sort_values(by='id', inplace=True)
res.reset_index(drop=True, inplace=True)
res.to_csv('/content/drive/MyDrive/Data/submission_all.csv', index=False)

# 展示预测结果

class_cancer = {0: 'benign', 1: 'malignant'}

fig, axes = plt.subplots(2, 2, figsize=(50, 30), facecolor='w')
plt.subplots_adjust(wspace=1, hspace=1)
all = 0
acc = 0
error = 0

for ax in axes.ravel():
    i = random.choice(res['id'].values)
    label = res.loc[res['id'] == i, 'label'].values[0]
    label = str(label).split('[')[1].split(']')[0]
    print(label)
    if os.path.exists('/content/drive/MyDrive/Data/new_BreaKHis/test/new_benign/' + str(i) + '.png'):
        path_b = '/content/drive/MyDrive/Data/new_BreaKHis/test/new_benign/' + str(i) + '.png'
        img = Image.open(path_b)
        ax.set_title('Ben-->' + class_cancer[label[0]])
        if 'benign' == class_cancer[label[0]]:
            acc += 1
            all += 1
        else:
            all += 1
            if os.path.exists('/content/drive/MyDrive/Data/new_BreaKHis/pred_error/ben_to_mal/' + str(i) + '.png'):
                error = error
            else:
                error += 1
                print('\n第{}张是错的，良性预测为恶性'.format(all))
                print('原地址为：', path_b)
                img.save('/content/drive/MyDrive/Data/new_BreaKHis/pred_error/ben_to_mal/' + str(i) + '.png')

        # ax.set_title(path_b+class_cancer[label[0]])
    else:
        path_m = '/content/drive/MyDrive/Data/new_BreaKHis/test/new_malignant/' + str(i) + '.png'
        img = Image.open(path_m)
        ax.set_title('Mal-->' + class_cancer[label[0]])
        if 'malignant' == class_cancer[label[0]]:
            acc += 1
            all += 1
        else:
            all += 1
            if os.path.exists('/content/drive/MyDrive/Data/new_BreaKHis/pred_error/mal_to_ben/' + str(i) + '.png'):
                error = error
            else:
                error += 1
                print('\n第{}张是错的,恶性预测为良性'.format(all))
                print('原地址为：', path_m)
                img.save('/content/drive/MyDrive/Data/new_BreaKHis/pred_error/mal_to_ben/' + str(i) + '.png')

        # ax.set_title(path_m+class_cancer[label[0]])

    ax.imshow(img)

plt.savefig('/content/drive/MyDrive/Data/this.png')

print('The predicted correct number is:{}, error number is:{}'.format(all - error, error))
print('all is:', all)
print('the acc is:{:.4f}%'.format((all - error) * 100 / all))
