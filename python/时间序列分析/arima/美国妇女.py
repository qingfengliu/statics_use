import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf,plot_predict
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.stats.diagnostic import acorr_lm

data=pd.read_csv(r'D:\\书籍资料整理\\时间序列分析_王燕\\file18.csv',index_col='year')
data.index=pd.to_datetime(data.index,format='%Y')
#绘出时序图
# plt.plot(data)
# plt.show(block=True)
#是一个明显不平稳的序列
# 1.进行平稳性检验DF检验、ADF检验、PP检验,参看多元时序一章的记录
t = adfuller(data,regression='ct')  # ADF检验
# 对于ADF延迟阶数怎么理解？
# lags为延迟期数，如果为整数，则是包含在内的延迟期数，如果是一个列表或数组，那么所有时滞都包含在列表中最大的时滞中
# 在别的函数中找到的解释不知道对不对
output = pd.DataFrame(index=['统计量值', "p值", "滞后数", "Number of Observations Used",
                             "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
output['value']['统计量值'] = t[0]
output['value']['p值'] = t[1]
output['value']['滞后数'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)#不能显著拒绝原假设,所以视为非平稳的

data['ts1']=data.diff(1)  #进行一次差分
t = adfuller(data['ts1'].dropna(),regression='c')  # ADF检验
output = pd.DataFrame(index=['统计量值', "p值", "滞后数", "Number of Observations Used",
                             "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
output['value']['统计量值'] = t[0]
output['value']['p值'] = t[1]
output['value']['滞后数'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)#这里ts1没有过,ADF检验,但是看时序图,较为平稳.有几个周期的方差较大怀疑是异方差

#做lm检验模型是显著的异方差,但是书中根据经验判断,适用arima模型.
#那么臆测一下,异方差由于历史的几个周期导致,因为数据比较历史.并且周期少
#那么可以认为适用arima,解决之后开始绘制自相关图和偏自相关图
c=acorr_lm(data['ts1'].dropna())
print(c)

# lag_acf = acf(data['ts1'].dropna(), nlags=10,fft=False)
# lag_pacf = pacf(data['ts1'].dropna(), nlags=10, method='ols')
# fig, axes = plt.subplots(1,2, figsize=(20,5))
# plot_acf(data['ts1'].dropna(), lags=10, ax=axes[0])
# plot_pacf(data['ts1'].dropna(), lags=10, ax=axes[1], method='ols')
# plt.show(block=True)

# 疏系数模型书中给出的是ARIMA((1,4),1,0)
# 但是我在看貌似ARIMA((1,4),1,1)更好些

# order_trend=arma_order_select_ic(data['ts1'].dropna())#这里由于异方差,可能没给出最好结果
# print(order_trend['bic_min_order'])

# python疏系数方法，对比了arima(4,1,0)和(4,1,1)后根据AIC和BIC使用(4,1,0)更好
result_trend=ARIMA(data['fertility'],order=(4,1,0),enforce_stationarity=False)
with result_trend.fix_params({'ar.L2':0,'ar.L3':0}):
    result_trend=result_trend.fit()
    print(result_trend.param_names)
    print(result_trend.forecast())

