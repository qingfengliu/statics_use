#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# Dataset 封装了__getitem__和__len__也是不可直接迭代的,需要使用
# DataLoader 迭代
class CIFAR10TrainDataset(Dataset):
    def __init__(self, file_path=[], crop_size_img=None, crop_size_label=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        labels = pd.read_csv(file_path[1])
        self.labels = labels['y'].to_list()[:49000]

    #         print("labelstrain:",len(self.labels))
    #         print(len(self.imgs),len(self.labels))

    def __getitem__(self, index):
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = self.imgs[index]
        img = Image.open(img)
        label = self.labels[index]
        img = self.img_transform(img, label)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        files_list = [x for x in files_list if
                      int(re.search(r'(?<=train\\)[0-9]+', os.path.join(path, x)).group(0)) <= 49000]
        #         print("datatrain:",len(files_list))
        file_path_list = {
            int(re.search(r'(?<=train\\)[0-9]+', os.path.join(path, img)).group(0)) - 1: os.path.join(path, img) for img
            in files_list}
        return file_path_list

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        transform = transforms.Compose(
            [
                # 在高度和宽度上将图像放大到40像素的正方形
                #             torchvision.transforms.Resize(40),
                torchvision.transforms.Resize((224, 224)),  # 是推荐的形状要试试
                # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
                # 生成一个面积为原始图像面积0.64到1倍的小正方形，
                # 然后将其缩放为高度和宽度均为32像素的正方形
                #             torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                #                                                            ratio=(1.0, 1.0)),
                #             torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                # 标准化图像的每个通道
                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                 [0.2023, 0.1994, 0.2010])
            ]
        )
        img = transform(img)

        return img


class CIFAR10TestDataset(Dataset):
    def __init__(self, file_path=[], crop_size_img=None, crop_size_label=None):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        labels = pd.read_csv(file_path[1])
        self.labels = labels['y'].to_list()[49000:]

    #         print("labelstest:",len(self.labels))

    def __getitem__(self, index):
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = self.imgs[index]
        img = Image.open(img)
        label = self.labels[index]
        img = self.img_transform(img, label)

        return img, label

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        files_list = [x for x in files_list if
                      int(re.search(r'(?<=train\\)[0-9]+', os.path.join(path, x)).group(0)) > 49000]
        #         print("datatest:",len(files_list))
        file_path_list = {
            int(re.search(r'(?<=train\\)[0-9]+', os.path.join(path, img)).group(0)) - 49001: os.path.join(path, img) for
            img in files_list}
        return file_path_list

    def img_transform(self, img, label):
        """对图片和标签做一些数值处理"""
        transform = transforms.Compose(
            [
                # 在高度和宽度上将图像放大到40像素的正方形
                #             torchvision.transforms.Resize(40),
                torchvision.transforms.Resize((224, 224)),  # 是推荐的形状要试试
                # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
                # 生成一个面积为原始图像面积0.64到1倍的小正方形，
                # 然后将其缩放为高度和宽度均为32像素的正方形
                #             torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                #                                                            ratio=(1.0, 1.0)),
                #             torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                # 标准化图像的每个通道
                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                 [0.2023, 0.1994, 0.2010])
            ]
        )
        img = transform(img)

        return img

def try_gpu(i=0):  # @save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, lr, device, net_dict, init_parma=True):
    """用GPU训练模型(在第六章定义)"""
    if init_parma:
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    # 定义优化器
    #     optimizer = torch.optim.SGD(net.parameters(), lr=lr) #21轮以后会循环
    optimizer = torch.optim.RMSprop(net.parameters(),
                                    lr=lr,
                                    alpha=0.99,
                                    eps=1e-08,
                                    weight_decay=0,
                                    momentum=0,
                                    centered=False)

    # 定义损失函数
    loss = nn.CrossEntropyLoss()

    num_batches = len(train_iter)
    # 迭代
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录当前时间
        # [训练损失之和，训练准确率之和，样本数]
        metric = Accumulator(3)
        # 在训练之前必加的,
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        net_dict.append({'echo': epoch + 1, 'test_acc': test_acc, 'params': net.state_dict()})
        if test_acc > 0.85:
            print("Early stopping")
            break
        print('epoch{} loss:{:.4f} train_acc:{:.4f} test_acc:{:.4f} time:{:.4f}'.format(epoch + 1, l.item(), train_acc,
                                                                                        test_acc,
                                                                                        time.time() - start_time))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')

dataset_train = CIFAR10TrainDataset([r'D:\书籍资料整理\kaggle\cifar-10\train', r'D:\书籍资料整理\kaggle\cifar-10\label.csv'])
dataset_test = CIFAR10TestDataset([r'D:\书籍资料整理\kaggle\cifar-10\train', r'D:\书籍资料整理\kaggle\cifar-10\label.csv'])
dataloader_train = DataLoader(dataset_train,  # 封装的对象
                              batch_size=16,  # 输出的batchsize
                              shuffle=True,  # 随机输出
                              num_workers=0)  # 只有1个进程
dataloader_test = DataLoader(dataset_test,  # 封装的对象
                             batch_size=16,  # 输出的batchsize
                             shuffle=False,
                             num_workers=0)  # 只有1个进程


print(torch.cuda.device_count())

net = torchvision.models.resnet50(pretrained=False)  # 使用resnet50模型[残差网络] 不进行预训练
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 10)

device = try_gpu()
net_dict = []
train(net, dataloader_train, dataloader_test, 50, 0.1, device, net_dict)



file_name = r'D:\书籍资料整理\kaggle\cifar-10\cifar10_resnet50_30.pt'
# torch.save(net, file_name)  #这种方法 使用不成功,网上说绑定到 什么类没看懂
torch.save(net_dict[35]['params'], file_name)
print(file_name + ' saved successfully!')



