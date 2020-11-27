import pandas as pd
import math
import statsmodels.api as sm
import numpy as np
#1.比例差、相对风险、优势比
data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/阿司匹林与心脏病.csv")
data=data.pivot_table(index="组",columns="心肌梗塞",values="值")
# print(data.loc['安慰剂','是'])

data["行总和"] =data.apply(lambda x:x.sum(),axis =1)
p1=float(data.loc['安慰剂','是']/data.loc['安慰剂','行总和'])
p2=float(data.loc['阿司匹林','是']/data.loc['阿司匹林','行总和'])

#(1)比例差
p1-p2  #比例差
se=math.sqrt(p1*(1-p1)/float(data.loc['安慰剂','行总和'])+p2*(1-p2)/float(data.loc['阿司匹林','行总和']))
se  #标准差
p1-p2-1.96*se,p1-p2+1.96*se#置信区间

#(2)相对风险和优势比,这里有包来做这件事了
# data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/阿司匹林与心脏病.csv")
p1/p2
table=sm.stats.Table.from_data(data)
np.asarray([data.loc[:,'是'],data.loc[:,'否']])
t22 = sm.stats.Table2x2(np.asarray([data.loc[:,'是'],data.loc[:,'否']]))
t22.summary()
#这个结果会给出优势比和置信区间
#这个结果给出了一个Risk ratio并不是相对风险.适用于对照组中的相对风险.
#而优势比是对的
#鉴于书中没有提到相对风险的置信区间求法先放在这了

#2.独立性检验
print(table.test_nominal_association())

