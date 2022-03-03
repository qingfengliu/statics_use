import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import torchvision

#loss 0.011, train acc 0.997, test acc 0.922
def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="D:/书籍资料整理/torch_data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="D:/书籍资料整理/torch_data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=2),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=2))

#-------------------------残差块与ResNet块-----------------------
#残差块
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

#-------------------------------------------------------------

#--------------------------训练部分函数-------------------------
def try_gpu(i=0):  #@save
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

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
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

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    #定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    #定义损失函数
    loss = nn.CrossEntropyLoss()

    num_batches = len(train_iter)
    #迭代
    for epoch in range(num_epochs):
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

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')

if __name__ == '__main__':
    blk = Residual(3,3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)  #输入与输出

    #我们也可以在增加输出通道数的同时，减半输出的高和宽。
    blk = Residual(3,6, use_1x1conv=True, strides=2)
    print(blk(X).shape)

    #ResNet 最底层块,由一个卷积层采样输入图片,然后经过批量归一化层.最大池化层
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    #这里为啥带星因为该层并不是由类创建而是由函数resnet_block函数返回是数组,
    #数组返回两个残差块函数resnet_block将其打包.使用*将其转换成类似类的那种形式,可以称为封装
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    #就剩下了尾部的半部分,AdaptiveAvgPool2d 相当于一个平均池化层
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(), nn.Linear(512, 10))
    #看下各层结构
    # X = torch.rand(size=(1, 1, 224, 224))
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__,'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())