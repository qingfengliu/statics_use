import numpy as np
#这里权重是2*4的张量
#b1是1*4的张量
#x是10*2的张量
#h是10*4的张量

W1=np.random.randn(2,4) #权重
b1=np.random.randn(4)
x=np.random.randn(10,2)
h=np.dot(x,W1)+b1

#赋予输出激活函数
#经过激活函数,输出的还是10*4的张量b
def sigmoid(x):
    return 1/(1+np.exp(-x))
a=sigmoid(h)
print(a)
