import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
iris=datasets.load_iris()

data=pd.read_csv(r"D:\书籍资料整理\多元统计分析\人类发展水平.csv")
#这个应该是贝叶斯判别
lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
x=data[['出生时预期寿命','预期受教育年限','平均受教育年限','人均公民总收入']]
x=np.asarray(x)
y=data['等级']
# print(x)
result=lda.fit(x,y)
# print(lda.predict(np.asarray([[84.2,15.7,11.6,54265],[76,13.5,7.6,13345]])))
lda.transform(x)

