import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import time
import matplotlib.pyplot as plt
data=pd.read_csv(r"D:\书籍资料整理\多元统计分析\表3-7.csv")
txt=data['国家']
#+','+data['标记']

train=data[['人均国内生产总值','粗死亡率','粗出生率','城镇人口比重','平均预期寿命','65岁以上人口比重']].apply(
    lambda x: (x - np.mean(x)) / (np.std(x)))
model = MDS(n_components=2)
model.fit(train)

temp=model.fit_transform(train)
# X_transformed = embedding.fit_transform(X[:100])
# X_transformed.shape
x=temp[:,0]
y=temp[:,1]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
plt.scatter(x, y)
for i in range(len(temp[:,0])):
    plt.annotate(txt[i], xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1)) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
plt.show()

#与书中的图不太相同,并且每次运行的结果点的位置都是不同的
#但是相对位置是相近的
#注意,如果要使用非度量方法,需要将MDS的dissimilarity参数设置为precomputed
#并且fit时x传递的是相似性矩阵