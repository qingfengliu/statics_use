import pandas as pd
import math
import statsmodels.api as sm
import numpy as np
from scipy import stats

#1.名义变量卡方统计量
data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/政党认同.csv")
tab = pd.crosstab(data['性别'], data['政党认同'],values=data['值'],aggfunc=np.sum)
# data=data.pivot_table(index="组",columns="心肌梗塞",values="值")


#但是书中政党认同的例子是给出了期望值,那么需要手动去求
statistic = (data['值']-data['期望']).apply(lambda x:x*x)/data['期望']

statistic=statistic.sum()
df = np.prod(np.asarray(tab.shape) - 1)
pvalue = 1 - stats.chi2.cdf(statistic, df)
statistic,pvalue

#2.有序统计量
#
data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/畸形与饮酒量.csv")
tab = pd.crosstab(data['饮酒量'], data['畸形'],values=data['值'],aggfunc=np.sum)
tab.reset_index(inplace=True)


tab['饮酒量']=tab['饮酒量'].replace({'0':0,'<1':0.5,'1-2':1.5,'3-5':4,'≥6':7})
tab.set_index('饮酒量',inplace=True)
tab.rename(columns={'否':0,'是':1},inplace=True)
tab.sort_index(inplace=True)

#当期望值没有给出时可以通过如下方法
table=sm.stats.Table(tab)
#使用test_ordinal_association要指定行列得分
table.test_ordinal_association(row_scores=np.array([0,0.5,1.5,4,7]),col_scores=np.array([0,1]))
#test_ordinal_association  这个检验是指行列变量都为有序变量

#3.小样本费舍尔品茶实验。留在这先不做了
