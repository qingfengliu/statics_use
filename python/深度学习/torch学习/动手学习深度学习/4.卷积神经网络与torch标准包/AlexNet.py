import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
import torchvision
import time
import numpy as np

#卷积网络的应用是比较复杂的网络,并且使用GPU,在学习的时候没有完全吃透代码,这里吃透
#首先比起,渐层的多层感知机代码,主要的还是增加了将数据传入GPU过程为了保持通用封装了函数
#卷积神经网络,在图片处理领域是十分成熟的,本章内容书要是实现,较为成熟的多层卷积模型.
#其实其思路差不多,最主要的是思考,落入各层的数据是什么样的这才是最主要的.

#其实这章的函数比原来的数据更少,因为load_data_fashion_mnist已经处理好比较干净的图片数据
#仅多了evaluate_accuracy_gpu,计算了对测试集精度
#

class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

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

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

#Xavier是一种参数初始化方式
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


#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    #定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    #定义损失函数
    loss = nn.CrossEntropyLoss()

    timer, num_batches = Timer(), len(train_iter)
    #迭代
    for epoch in range(num_epochs):
        # [训练损失之和，训练准确率之和，样本数]
        metric = Accumulator(3)
        # 在训练之前必加的,
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

        test_acc = evaluate_accuracy_gpu(net, test_iter)

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

#ImageNet使用的模型但是本文中使用在了fashion mnist中,将原有fashion mnist拓展为224*224
#第一层为11*11步幅为4填充为1的图片,所以输出为54这里我是这样计算的,以一个维度为例,224加上填充为226然后kernel_size=11
#那么必有(11-1)/2*2也就是10个像素被舍弃,那么变为216,由于步进为4那么216/4=54
#最大池化层3步进为2,那么输出为26*26,维度为96
#第二个卷积层将图片通道数变为256,且长宽没变,
#然后池化层将图片变为12*12
#接下来的池化层,并不改变图片形状只是改变了通道数
#那么有个问题,卷积层是如何改变通道数的最后的卷积层形状为256通道12*12的图片
#然后卷积层图片变为5*5
#Linear展平为256*5*5
if __name__ == "__main__":
    net = nn.Sequential(
        # 这里，我们使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10))


    batch_size = 128
    #AlexNet是在ImageNet上进行训练的
    #这里使用的是Fashion-MNIST数据集,因为在ImageNet训练时间会很长
    #需要将Fashion-MNIST放大
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.01, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())