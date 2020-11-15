import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from arch.univariate import ARCH, GARCH
data=pd.read_csv(r'D:\书籍资料整理\时间序列分析_王燕\file22.csv',index_col='month')

data.index=pd.to_datetime(data.index,format='%Y/%m/%d')

#绘出时序图
# plt.plot(data)
# plt.show(block=True)
#可以看出很典型的异方差

#此函数默认会给出一个GARCH(1,1)的模型
temp=arch_model(data).fit()
print(temp.arch_lm_test(lags=4))
# temp.hedgehog_plot(type='mean')
# plt.show(block=True)
temp_forecast=temp.forecast(horizon=5)
print(temp_forecast.variance)
#感觉这个例子无论是书中还是哪里都感觉非常乱,暂时不研究了

