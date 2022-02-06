import torch
from torch import nn

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
print(torch.cuda.device_count())
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())
#cuda就是gpu的驱动

#在gpu上创建张量
X = torch.ones(2, 3, device=try_gpu())
print(X)

#因为仅有1个GPU
Y = torch.rand(2, 3, device=try_gpu(1))
print(Y)

#X本在GPU上,这里书中本意是将X复制到GPU1上,
#然后本设备仅有1个GPU,所以复制到了CPU上
#Z = X.cuda(1)  #这里会报错因为没有cuda1
Z = X.to('cpu')
print(X)
print(Z)

print(Y+Z)

#将模型保存到GPU上
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)
