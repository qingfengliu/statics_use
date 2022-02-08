#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

x = torch.arange(12)

# In[ ]:


x

# In[ ]:


x.shape

# In[ ]:


x.numel()

# In[ ]:


torch.zeros((2, 3, 4))  # torch张量是从外到内包装的.   0张量

# In[ ]:


torch.ones((2, 3, 4))  # 1张量

# In[ ]:


torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# In[ ]:


x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # torch运算符,升级成按元素运算

# In[ ]:


# 张量连接
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

# In[ ]:


X == Y  # 张量的逻辑运算符

# In[ ]:


before = id(Y)
Y = Y + X
id(Y) == before

# In[ ]:


before = id(X)
X += Y
id(X) == before

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from IPython import display
import matplotlib.pyplot as plt


def use_svg_display():  # @save
    """使用svg格式在Jupyter中显示绘图"""
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):  # @save
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# @save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# @save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


def f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])

# In[ ]:


# 自动微分
x = torch.arange(4.0, requires_grad=True)
x.grad  # 默认值是None
y = 2 * torch.dot(x, x)

# In[ ]:


# x是一个长度为4的向量，计算x和x的点积，得到了我们赋值给y的标量输出。
# 接下来，我们通过调用反向传播函数来自动计算y关于x每个分量的梯度，并打印这些梯度。
# 可以这样理解,对于torch的向量,进行运算,torch会记录运算方式,并且,可随时反向传播
y.backward()
x.grad

# In[ ]:


x.grad == 4 * x

# In[ ]:


# 计算另外一种梯度
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad

# In[ ]:


# 分离计算，有时，我们希望将某些计算移动到记录的计算图之外。 例如，假设y是作为x的函数计算的，而z则是作为y和x的函数计算的。
# 想象一下，我们想计算z关于x的梯度，但由于某种原因，我们希望将y视为一个常数， 并且只考虑到x在y被计算后发挥的作用。
# 在这里，我们可以分离y来返回一个新变量u，该变量与y具有相同的值， 但丢弃计算图中如何计算y的任何信息。
# 换句话说，梯度不会向后流经u到x。 因此，下面的反向传播函数计算z=u*x关于x的偏导数，同时将u作为常数处理，
# 而不是z=x*x*x关于x的偏导数。
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u

# In[ ]:


# 由于记录了y的计算结果，我们可以随后在y上调用反向传播，
# 得到y=x*x关于的x的导数，即2*x
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from torch.distributions import multinomial

# fair_probs 是骰子 各个点数的概率
# 将概率给多项式分布,并采样一个点
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()

# In[ ]:


# 这个多项分布生成应该有多个版本在torch上.
# 搜错版本了,第一个参数是,试验次数,
# 第二个参数是一个向量,向量值为概率,长度为允许出现的值
multinomial.Multinomial(10, fair_probs).sample()

# In[ ]:


# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # 相对频率作为估计值

# In[ ]:




