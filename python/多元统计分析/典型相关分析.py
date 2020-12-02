import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
#sklearn 的CCA包主要是用于交叉分解的,所以其根本目的不在于探究
#典型相关性
data=pd.read_csv(r"D:\书籍资料整理\多元统计分析\例8-2.csv")
X=data[['地区生产总值','第二产业增加值','第三产业增加值','人均地区生产总值','生产总值增长率','固定资产投资']]
Y=data[['PM2.5','PM10','SO2','CO','NO2','O3']]
cca = CCA(n_components=1)
cca.fit(X, Y)

X_H,Y_H=cca.transform(X,Y)


X_H=X_H.T
Y_H=Y_H.T
X_H=list(X_H)[0]
Y_H=list(Y_H)[0]

#这里得到了第一个典型方程的相关系数
temp=pd.DataFrame({'x':X_H,'y':Y_H})
temp.corr()
#如果数据是两个参数U1,U2.那么X_H 返回U1,U2
#Y_H返回V1,V2
#下边这个式子是权重但是不奇怪它做了某种处理
#使与R的不同
cca.x_weights_

