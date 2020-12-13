import numpy as np
import pandas as pd
from sklearn import  linear_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#1.岭回归
data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/表6-1.csv")
x=data[['x1','x2','x3','x4','x5']]
y=data[['y']]
scaler = StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
scaler = StandardScaler()
scaler.fit(y)
y=scaler.transform(y)


#书中数据是将x和y标准化归一化后的数据
clf = linear_model.Ridge(alpha=0.05,solver='svd',normalize=True)
clf.fit(x, y)
#各β

alphas = np.arange(0.1,0.3,0.02)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, solver='svd',normalize=True)
    ridge.fit(x, y)
    #这里与手册的例子不一样,因为在标准化的时候改变了数据结构
    coefs.append(ridge.coef_[0])

ax = plt.gca()
ax.plot(alphas, coefs)

# ax.set_xlim(ax.get_xlim()[::-1])  # x轴会翻转
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
#使用sklearn做岭回归不会显示那么多,参数统计量

#2.主成分回归。就是先做PCA然后再进行回归,在sklearn中将数据PCA后转换,然后做线性回归后,将主成分带入原方程中即可.

