#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import pandas as pd
import os
import re
from PIL import Image
from torchvision.models import resnet34
import time
import joblib


def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class CIFAR10PreDataset(Dataset):
    def __init__(self, file_path=[], crop_size_img=None, crop_size_label=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        self.img_path = file_path[0]
        self.imgs = self.read_file(self.img_path)

    def __getitem__(self, index):
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = self.imgs[index]
        img = Image.open(img)

        img = self.img_transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = {
            int(re.search(r'(?<=test\\)[0-9]+', os.path.join(path, img)).group(0)) - 1: os.path.join(path, img) for img
            in files_list}
        return file_path_list

    def img_transform(self, img, ):
        """对图片和标签做一些数值处理"""
        transform = transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                transforms.ToTensor(),
                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                 [0.2023, 0.1994, 0.2010])
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform(img)

        return img

net = torchvision.models.resnet50(pretrained=False)  # 使用resnet50模型[残差网络] 不进行预训练
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 10)
net.load_state_dict(torch.load(r'D:\书籍资料整理\kaggle\cifar-10\cifar10_resnet50_0.84.pt'))

device = try_gpu()
net.to(device)
net.eval()  # 切换到测试模式

dataset_pre = CIFAR10PreDataset([r'D:\书籍资料整理\kaggle\cifar-10\test'])

dataloader_pre = DataLoader(dataset_pre,  # 封装的对象
                            batch_size=1,  # 输出的batchsize
                            shuffle=False,  # 不需要随机数出
                            num_workers=0)  # 只有1个进程

labels = joblib.load(r'D:\书籍资料整理\kaggle\cifar-10\label.pkl')
labels = labels.classes_
pre_data = pd.DataFrame(columns=['id', 'label'])
for index, data in enumerate(dataloader_pre):
    data = data.to(device)
    pre_label = net(data)
    _, preds = torch.max(pre_label, 1)

    pre_data = pre_data.append({'id': index + 1, 'label': labels[preds.item()]}, ignore_index=True)



pre_data.head()


pre_data.to_csv(r'D:\书籍资料整理\kaggle\cifar-10\jieguo.csv', index=False)


