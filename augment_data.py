import os
from time import sleep

import cv2
import numpy as np
from random import choice
import random


def Image_rotate(img):
    """
    :参数img:原始图片矩阵
    :返回值:旋转中心，旋转角度，缩放比例
    """
    rows, cols = img.shape[:2]
    rotate_core = (cols / 2, rows / 2)
    rotate_angle = [-180, 180, 45, -45, 90, -90, 210, 270, -210, -270]
    paras = cv2.getRotationMatrix2D(rotate_core, choice(rotate_angle), 1)
    border_value = tuple(int(x) for x in choice(choice(img)))
    img_new = cv2.warpAffine(img, paras, (cols, rows), borderValue=border_value)
    return img_new


def Image_traslation(img):
    """
    :参数img: 原始图片矩阵
    :返回值: [1, 0, 100]-宽右移100像素； [0, 1, 100]-高下移100像素
    """
    paras_wide = [[1, 0, 100], [1, 0, -100]]
    paras_height = [[0, 1, 100], [0, 1, -100]]
    rows, cols = img.shape[:2]
    img_shift = np.float32([choice(paras_wide), choice(paras_height)])
    border_value = tuple(int(x) for x in choice(choice(img)))
    img_new = cv2.warpAffine(img, img_shift, (cols, rows), borderValue=border_value)
    return img_new


# 癌症图像能否加噪声
def Image_noise(img):
    """
    :param img:原始图片矩阵
    :return: 0-高斯噪声，1-椒盐噪声
    """
    paras = [0, 1]
    gaussian_class = choice(paras)
    noise_ratio = [0.05, 0.06, 0.08]
    if gaussian_class == 1:
        output = np.zeros(img.shape, np.uint8)
        prob = choice(noise_ratio)
        thres = 1 - prob
        # print('prob', prob)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        return output
    else:
        mean = 0
        var = choice([0.001, 0.002, 0.003])
        # print('var', var)
        img = np.array(img / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, img.shape)
        out = img + noise
        if out.min() < 0:
            low_clip = -1
        else:
            low_clip = 0
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        return out

    """
    path_read: 读取原始数据集图片的位置;
    path_write：图片扩增后存放的位置；
    picture_size：图片之后存储的尺寸;
    enhance_num: 需要通过扩增手段增加的图片数量
    """


# 路径列表
# D:/python-Project/cancer_detection/BreaKHis 400X/train/benign
# D:/python-Project/cancer_detection/BreaKHis 400X/train/malignant
# D:/python-Project/cancer_detection/BreaKHis 400X/test/benign
# D:/python-Project/cancer_detection/BreaKHis 400X/test/malignant
# D:/python-Project/cancer_detection/BreaKHis 400X/val/benign
# D:/python-Project/cancer_detection/BreaKHis 400X/val/malignant

# D:/python-Project/cancer_detection/new_BreaKHis/train/benign
# D:/python-Project/cancer_detection/new_BreaKHis/train/malignant
# D:/python-Project/cancer_detection/new_BreaKHis/test/benign
# D:/python-Project/cancer_detection/new_BreaKHis/test/malignant
# D:/python-Project/cancer_detection/new_BreaKHis/val/benign
# D:/python-Project/cancer_detection/new_BreaKHis/val/malignant

# 原良性图像,原恶性图像,训练集
# 扩增训练集
def augment_train_benign(path_read, path_write):
    path_read_benign = path_read
    image_list_benign = [x for x in os.listdir(path_read_benign)]
    path_write_benign = path_write

    existed_img = len(image_list_benign)
    print(f"训练集良性图像数量为 {existed_img}")
    enhance_num = existed_img * 6
    while enhance_num > 0:
        if enhance_num > existed_img * 2:
            img = choice(image_list_benign)
            image = cv2.imread(path_read_benign + '/' + img, cv2.IMREAD_COLOR)
            image = Image_rotate(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_benign + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
        else:
            img = choice(image_list_benign)
            image = cv2.imread(path_read_benign + '/' + img, cv2.IMREAD_COLOR)
            image = Image_traslation(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_benign + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
    image_list_new_benign = [x for x in os.listdir(path_write_benign)]
    new_existed_img = len(image_list_new_benign)
    print(f"扩充训练集良性图像数量为 {new_existed_img}")


def augment_train_malignant(path_read, path_write):
    path_read_malignant = path_read
    image_list_malignant = [x for x in os.listdir(path_read_malignant)]
    path_write_malignant = path_write
    existed_img = len(image_list_malignant)
    print("训练集恶性图像数量为 %d" % (existed_img))
    enhance_num = existed_img * 6
    while enhance_num > 0:
        if enhance_num > existed_img * 2:
            img = choice(image_list_malignant)
            image = cv2.imread(path_read_malignant + '/' + img, cv2.IMREAD_COLOR)
            image = Image_rotate(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_malignant + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
        else:
            img = choice(image_list_malignant)
            image = cv2.imread(path_read_malignant + '/' + img, cv2.IMREAD_COLOR)
            image = Image_traslation(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_malignant + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
    image_list_new_malignant = [x for x in os.listdir(path_write_malignant)]
    new_existed_img = len(image_list_new_malignant)
    print(f"扩充训练集恶性图像数量为 {new_existed_img}")


# 扩增测试集
def augment_test_benign(path_read, path_write):
    path_read_benign = path_read
    image_list_benign = [x for x in os.listdir(path_read_benign)]
    path_write_benign = path_write
    image_list_new_benign = [x for x in os.listdir(path_write_benign)]
    existed_img = len(image_list_benign)
    print("测试集良性图像数量为 %d" % (existed_img))
    enhance_num = existed_img * 4
    while enhance_num > 0:
        if enhance_num > existed_img:
            img = choice(image_list_benign)
            image = cv2.imread(path_read_benign + '/' + img, cv2.IMREAD_COLOR)
            image = Image_rotate(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_benign + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
        else:
            img = choice(image_list_benign)
            image = cv2.imread(path_read_benign + '/' + img, cv2.IMREAD_COLOR)
            image = Image_traslation(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_benign + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1

    new_existed_img = len(image_list_new_benign)
    print(f"扩充测试集良性图像数量为 {new_existed_img}")


def augment_test_malignant(path_read, path_write):
    path_read_malignant = path_read
    image_list_malignant = [x for x in os.listdir(path_read_malignant)]
    path_write_malignant = path_write
    existed_img = len(image_list_malignant)
    print(f"测试集恶性图像数量为 {existed_img}")
    enhance_num = existed_img * 4
    while enhance_num > 0:
        if enhance_num > existed_img:
            img = choice(image_list_malignant)
            image = cv2.imread(path_read_malignant + '/' + img, cv2.IMREAD_COLOR)
            image = Image_rotate(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_malignant + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
        else:
            img = choice(image_list_malignant)
            image = cv2.imread(path_read_malignant + '/' + img, cv2.IMREAD_COLOR)
            image = Image_traslation(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_malignant + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
    image_list_new_malignant = [x for x in os.listdir(path_write_malignant)]
    new_existed_img = len(image_list_new_malignant)
    print(f"扩充测试集恶性图像数量为 {new_existed_img}")


# 扩增验证集，11倍
def augment_val_benign(path_read, path_wirte):
    path_read_benign = path_read
    path_write_benign = path_wirte
    image_list_benign = [x for x in os.listdir(path_read_benign)]
    existed_img = len(image_list_benign)
    print("验证集良性图像数量为 %d" % (existed_img))
    enhance_num = existed_img * 10
    while enhance_num > 0:
        if enhance_num > existed_img * 3:
            img = choice(image_list_benign)
            image = cv2.imread(path_read_benign + '/' + img, cv2.IMREAD_COLOR)
            image = Image_rotate(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_benign + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
        else:
            img = choice(image_list_benign)
            image = cv2.imread(path_read_benign + '/' + img, cv2.IMREAD_COLOR)
            image = Image_traslation(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_benign + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
    image_list_new_benign = [x for x in os.listdir(path_write_benign)]
    new_existed_img = len(image_list_new_benign)
    print(f"扩充验证集良性图像数量为{new_existed_img}")


# 11倍
def augment_val_malignant(path_read, path_wirte):
    path_read_malignant = path_read
    image_list_malignant = [x for x in os.listdir(path_read_malignant)]
    existed_img = len(image_list_malignant)
    print(f"验证集恶性图像数量为 {existed_img}")
    path_write_malignant = path_wirte
    enhance_num = existed_img * 11
    while enhance_num > 0:
        if enhance_num > existed_img * 3:
            img = choice(image_list_malignant)
            image = cv2.imread(path_read_malignant + '/' + img, cv2.IMREAD_COLOR)
            image = Image_rotate(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_malignant + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
        else:
            img = choice(image_list_malignant)
            image = cv2.imread(path_read_malignant + '/' + img, cv2.IMREAD_COLOR)
            image = Image_traslation(image)
            newname = img.split('.')[0]
            newid = enhance_num - 1
            image_dir = path_write_malignant + '/' + 'NEW_' + str(newid) + '_' + newname + '.png'
            cv2.imwrite(image_dir, image)
            enhance_num -= 1
    image_list_new_malignant = [x for x in os.listdir(path_write_malignant)]
    new_existed_img = len(image_list_new_malignant)
    print(f"扩充验证集恶性图像数量为 {new_existed_img}")


# 验证集460张，测试集545张，
# 打印每个文件夹中的文件数量
def show_num(path):
    image_list = [x for x in os.listdir(path)]
    existed_img = len(image_list)
    print(f"图像数量为 {existed_img}")


show_num('D:/python-Project/cancer_detection/new_BreaKHis/train/malignant')
# augment_train_benign('D:/python-Project/cancer_detection/BreaKHis 400X/train/benign',
#                      'D:/python-Project/cancer_detection/new_BreaKHis/train/benign')
# augment_train_malignant('D:/python-Project/cancer_detection/BreaKHis 400X/train/malignant',
#                         'D:/python-Project/cancer_detection/new_BreaKHis/train/malignant')
# augment_test_benign('D:/python-Project/cancer_detection/BreaKHis 400X/test/benign',
#                     'D:/python-Project/cancer_detection/new_BreaKHis/test/benign')
# augment_test_malignant('D:/python-Project/cancer_detection/BreaKHis 400X/test/malignant',
#                         'D:/python-Project/cancer_detection/new_BreaKHis/test/malignant')
# augment_val_benign('D:/python-Project/cancer_detection/BreaKHis 400X/val/benign',
#                    'D:/python-Project/cancer_detection/new_BreaKHis/val/benign')
# augment_val_malignant('D:/python-Project/cancer_detection/BreaKHis 400X/val/malignant',
#                       'D:/python-Project/cancer_detection/new_BreaKHis/val/malignant')
