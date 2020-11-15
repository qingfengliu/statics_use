#使用包是可以进行X11的但是其步骤没有抽象到一个类别里
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.stattools import adfuller

def draw_ts(timeSeries,title):
    f = plt.figure(facecolor = "white")
    timeSeries.plot(color = "blue")
    plt.title(title)
    plt.show()

data=pd.read_excel(r'D:\书籍资料整理\时间序列分析_王燕\A1_17.xlsx',index_col='time')
data.index=pd.to_datetime(data.index,format='%Y/%m/%d')
#绘出时序图
# plt.plot(data)
# plt.show(block=True)

#1.确定是乘法模型还是加法模型,这里直接确定是乘法模型

decomposition = seasonal_decompose(data, model="multiplicative")
trend=decomposition.trend  #趋势分量
seasonal_arr = decomposition.seasonal   #季节分量
residual = decomposition.resid      #随机分量
trend = trend.dropna()
residual = residual.dropna()
residual_mean=np.mean(residual.values)
# draw_ts(data['x'], 'origin')
# draw_ts(trend, 'trend')
# draw_ts(seasonal, 'seasonal')
# draw_ts(residual, 'residual')

#2.处理趋势分量,(1)差分判断阶数。
t = adfuller(trend)  # ADF检验
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

# print(output) #可从时序图看出序列并非平稳的,并且这里检验也验证.

#(2)进行1阶差分
ts1=trend.diff(1)

t = adfuller(ts1.dropna())  # ADF检验
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
print(output) #依然非平稳

#(2)进行二阶差分
ts2=ts1.diff(1)
t = adfuller(ts2.dropna())  # ADF检验
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
# print(output) #二阶差分后序列平稳,p值很小

#可以确定ARIMA差分次数为2，
#(3)确定p,q阶数,这里就不画相关图和自相关图了

order_trend=arma_order_select_ic(ts2.dropna())

result_trend=ARIMA(trend,order=(1,2,1)).fit()
#可以确定使用ARMA(1,1)
#所以综上使用的模型是ARIMA(1,2,1)

diff_recover=result_trend.predict(typ ='levels')


recover = diff_recover['1993-03-01':'2000-12-01'] * seasonal_arr['1993-03-01':'2000-12-01'] * residual_mean

ts_quantum=data['1993-03-01':'2000-12-01']
recover=recover.dropna().to_frame()
recover.index.name='day'
ts_quantum.index.name='day'
recover.rename(columns={0:"yuce"},inplace=True)
ts_quantum.rename(columns={"x":"shiji"},inplace=True)

recover=pd.merge(recover,ts_quantum,how='left',left_index=True,right_index=True)
print(recover)
plt.figure(facecolor = "white")
recover.plot()
plt.show()