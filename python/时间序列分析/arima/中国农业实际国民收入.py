#这里会出现很多报错,是因为statsmodels包最新的很多函数的位置改变了,有空给调节回来
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf,plot_predict
from statsmodels.tsa.stattools import arma_order_select_ic


data=pd.read_csv(r'D:\书籍资料整理\时间序列分析_王燕\file17.csv',index_col='year')
data.index=pd.to_datetime(data.index,format='%Y')
#绘出时序图

# plt.plot(data)
# plt.show(block=True)
#1.进行平稳性检验DF检验、ADF检验、PP检验,参看多元时序一章的记录
#典型的非稳定的序列
# 1.进行平稳性检验DF检验、ADF检验、PP检验,参看多元时序一章的记录
t = adfuller(data)  # ADF检验
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
print(output)
#p值很大,可以接受原假设序列非平稳

#2.进入推断差分次数流程


data['ts1']=data.diff(1)  #进行一次差分
t = adfuller(data['ts1'].dropna())  # ADF检验
output = pd.DataFrame(index=['统计量值', "p值", "滞后数", "Number of Observations Used",
                             "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
output['value']['统计量值'] = t[0]
output['value']['p值'] = t[1]
output['value']['滞后数'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)#p值很小可以看做平稳序列了
#进行白噪声检验
output2=acorr_ljungbox(data['ts1'].dropna(),boxpierce=True,lags=[6,12],return_df=True)
print(output2)
#白噪声显示差分后的序列存在一些相关性可用arma模型


#3.求自相关系数和偏自相关系数
lag_acf = acf(data['ts1'].dropna(), nlags=10,fft=False)
lag_pacf = pacf(data['ts1'].dropna(), nlags=10, method='ols')
# fig, axes = plt.subplots(1,2, figsize=(20,5))
# plot_acf(data['ts1'].dropna(), lags=10, ax=axes[0])
# plot_pacf(data['ts1'].dropna(), lags=10, ax=axes[1], method='ols')
# plt.show(block=True)

order_trend=arma_order_select_ic(data['ts1'].dropna())
print(order_trend['bic_min_order']) #这里的选择和书中的一样

#4.拟合
result_trend=ARIMA(data['index'],(0,1,1)).fit()


print(result_trend.params)

#后边的步骤其实和ARMA一样了
#5.模型拟合度检验
#(1)残差的白噪声检验
output3=acorr_ljungbox(result_trend.resid,boxpierce=True,lags=[6,12],return_df=True)
print(output3)
#拟合后的白噪声检测效果很好,充分的大于了0.05

#(2)模型参数的显著性检验
print(result_trend.pvalues)#这个结果貌似与R的不太一致

fig, ax = plt.subplots()
ax = data['index'].loc['1952':].plot(ax=ax)
result_trend.plot_predict('1989','1992', dynamic=True,ax=ax,plot_insample=False)

plt.show()