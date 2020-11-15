from scipy.stats import binom
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from scipy.stats import t

def inom():
    #二项分布的概率分布图
    fig,ax = plt.subplots(1,1)
    n = 100
    p = 0.5
    #平均值, 方差, 偏度, 峰度
    mean,var,skew,kurt = binom.stats(n,p,moments='mvsk')
    # print(mean,var,skew,kurt)
    #ppf:累积分布函数的反函数。q=0.01时，ppf就是p(X<x)=0.01时的x值。
    #这一步是构建累计图的横坐标,当然不这么做
    x = np.arange(binom.ppf(0.01, n, p),binom.ppf(0.99, n, p))
    # x=np.arange(0,100)
    binom.pmf(x, n, p)
    # ax.plot(x, binom.pmf(x, n, p),'o')
    # plt.show()

    #1次试验成功的概率是0.06,没有不合格品概率
    print(binom.pmf(0,5,0.06))

    #5次试验成功的概率是0.06,没一个不合格品概率
    print(binom.pmf(1,5,0.06))

    #有3个及3个一下不合格品
    print(binom.cdf(3, 5, 0.06))

def zhengtai():
    fig, ax = plt.subplots(1, 1)
    mean, var, skew, kurt = norm.stats(moments='mvsk')

    x = np.linspace(norm.ppf(0.01),
                    norm.ppf(0.99), 100)
    x=np.append(x,np.linspace(norm.ppf(0.01,loc=1,scale=1),norm.ppf(0.99,loc=1,scale=1), 100))
    x=np.unique(x,axis=0)
    ax.plot(x, norm.pdf(x),'r-', lw=5, alpha=0.6, label='norm pdf')
    ax.plot(x, norm.pdf(x,loc=1,scale=1),'r-', lw=2, alpha=0.6, label='norm pdf')
    plt.show()
    #X＜40,
    norm.cdf(40,loc=50,scale=10)
    #累计概率为0.025时的反函数
    norm.ppf(0.025,loc=0,scale=1)


def kafang():
    #卡方分布仅有一个参数还是比较好理解的
    fig, ax = plt.subplots(1, 1)
    df = 5
    mean, var, skew, kurt = chi2.stats(df, moments='mvsk')
    x = np.linspace(chi2.ppf(0.01, df),
                    chi2.ppf(0.99, df), 100)
    ax.plot(x, chi2.pdf(x, df),
           'r-', lw=5, alpha=0.6, label='chi2 pdf')
    rv = chi2(df)
    # ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    # plt.show()

    #自由度为15,卡方值小于10的概率
    chi2.cdf(10,df=15)

    #卡方分布右尾概率为0.05时的反函数

    chi2.ppf(0.95,df=10)

def tfenbu():
    fig, ax = plt.subplots(1, 1)
    df = 5
    mean, var, skew, kurt = t.stats(df, moments='mvsk')
    x = np.linspace(t.ppf(0.01, df),
                    t.ppf(0.99, df), 100)
    # ax.plot(x, t.pdf(x, df),'r-', lw=5, alpha=0.6, label='t pdf')
    # plt.show()

    #自由度为10,t值小于-2的概率
    t.cdf(-2,df=10)
    #自由度为25,t分布右尾概率为0.025时的值
    t.ppf(0.975,df=25)