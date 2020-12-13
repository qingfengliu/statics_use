import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/第二章-一元回归分析.csv")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
#绘制x,y的点图
# plt.scatter(data['人均收入'], data['人均支出'])
# plt.show()



model=smf.ols('人均支出~人均收入',data=data)
result=model.fit()
#summary中提供了,决定系数
#并提供了F统计量。对于拟合优度检验来说足够
#并提供了参数和参数的范围
result.summary()

#Durbin-Watson证明残差非正态
#Jarque-Bera实际是样本是否遵循正态性状的检验,
#通过检验p=0,271没证据说明不服从正态.

res=data
#从统计量结果看残差有自相关性.
res['residual']=result.resid
plt.scatter(res['人均收入'], res['residual'])
plt.show()