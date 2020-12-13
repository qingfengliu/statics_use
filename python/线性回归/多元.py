import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/第三章-多元回归分析.csv")


model=smf.ols('y~x1+x2+x3+x4+x5',data=data)
result=model.fit()
#summary中提供了,决定系数
#并提供了F统计量。对于拟合优度检验来说足够
#并提供了参数和参数的范围
#通过结果可以看出，整体模型显著,但是每个变量的模型并不是很显著.

result.summary()
#相关阵可看出x1,x2,x3,x4,x5之间有很高的相关性
#存在共线性。
#如logistic,多重共线性会对变量的符号产生影响,显示出与显示不一样的问题。
data[['y','x1','x2','x3','x4','x5']].corr()



