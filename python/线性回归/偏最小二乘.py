import pandas as pd
import numpy as np
from sklearn import  linear_model
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/表8-1.csv")

x=data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']]
y=data[['y']]
scaler = StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
scaler = StandardScaler()
# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(x)
# x=min_max_scaler.transform(x)

scaler.fit(y)
y=scaler.transform(y)
# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(y)
# y=min_max_scaler.transform(y)
dy=[]
for i in y:
    dy.append(i[0])
y=np.asarray(dy)

#系数有误差
#scale=True代表归1话
#归1的方法是(x-x.mean)/x.std
pls2 = PLSRegression(n_components=3,scale=True)
pls2.fit(x, y)
print(pls2.coef_)

y_intercept = pls2.y_mean_ - np.dot(pls2.x_mean_ , pls2.coef_)
print(y_intercept)
# Y_pred = pls2.predict(X)