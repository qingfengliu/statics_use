
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf,plot_predict
from statsmodels.tsa.stattools import arma_order_select_ic


data=pd.read_csv(r'D:\书籍资料整理\时间序列分析_王燕\file8.csv',index_col='year')
data.index=pd.to_datetime(data.index,format='%Y')
#绘出时序图
plt.plot(data)
plt.show(block=True)
#1.进行平稳性检验DF检验、ADF检验、PP检验,参看多元时序一章的记录
t = adfuller(data)  # ADF检验
#对于ADF延迟阶数怎么理解？
#lags为延迟期数，如果为整数，则是包含在内的延迟期数，如果是一个列表或数组，那么所有时滞都包含在列表中最大的时滞中
#在别的函数中找到的解释不知道对不对
output=pd.DataFrame(index=['统计量值', "p值", "滞后数", "Number of Observations Used",
                           "Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['统计量值'] = t[0]
output['value']['p值'] = t[1]
output['value']['滞后数'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)

#可以看出p值很小,认为是稳定序列
#2.进行白噪声检验

output2=acorr_ljungbox(data.kilometer,boxpierce=True,lags=[6,12],return_df=True)
print(output2)
#6,12阶延迟均原小于0.05可以认为非白噪声

#3.下面可以开始建模过程,
#(1)观察自相关图和偏自相关图,定阶

#求自相关,偏自相关系数
lag_acf = acf(data, nlags=20,fft=False)
lag_pacf = pacf(data, nlags=20, method='ols')

#用自相关、偏自相关
fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(data, lags=20, ax=axes[0])
plot_pacf(data, lags=20, ax=axes[1], method='ols')
plt.show(block=True)

#这里python已经写了一个可以帮助选阶的函数
order_trend=arma_order_select_ic(data)
print(order_trend['bic_min_order'])
#结果为(2, 0)  也就是使用AR(2)模型


result_trend=ARMA(data,(2,0)).fit()
print(result_trend.params)
exit()
#result_trend.arparams   关于AR的参数
#result_trend.bic        BIC信息准则值
#result_trend.bse        参数的标准误
#result_trend.hqic       HQ信息准则
#result_trend.k_ar       AR系数的数量
#result_trend.k_ma       MA系数数量
#result_trend.k_trend    有常数时是1,没有常数时是0
#result_trend.llf        对数似然函数值
#result_trend.maparams   MA参数值
#result_trend.nobs       拟合所用观察数
#result_trend.params     模型的参数,顺序是趋势系数,k_exog外生系数,ar系数,ma系数。
#应该使用params来查看结果更好些
#result_trend.pvalues    系数的p值,基于的是z统计量
#result_trend.resid      模型残差
#result_trend.sigma2     残差的方差

#4.模型拟合度检验
#(1)残差的白噪声检验
output3=acorr_ljungbox(result_trend.resid,boxpierce=True,lags=[6,12],return_df=True)
print(output3)

#(2)模型参数的显著性检验
print(result_trend.pvalues)#这个结果貌似与R的不太一致


fig, ax = plt.subplots()
ax = data.loc['1950':].plot(ax=ax)
result_trend.plot_predict('2009','2012', dynamic=True,ax=ax,plot_insample=False)

plt.show()
