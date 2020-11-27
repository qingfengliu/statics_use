#配对数据关联样本
#1.McNemar检验
import pandas as pd
import numpy as np
from statsmodels.sandbox.stats.runs import mcnemar
from  statsmodels.stats.contingency_tables import SquareTable
from statsmodels.stats.inter_rater import cohens_kappa
from statsmodels.discrete.conditional_models import (
      ConditionalLogit, ConditionalPoisson, ConditionalMNLogit)


#1.检验量
#(1)对称性检验(齐性检验)
data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/环保.csv")

temp=np.array([[227,132],[107,678]])
#使用此种参数与书中结果最接近。这个应该是书中提到的方形列联表的检测方法.
#其结果与那个说的更相近
result = mcnemar(temp,exact=False,correction=False)

tmp=pd.DataFrame()
for i in range(0,4):
    tmp=tmp.append([data.loc[i]]*data.iloc[i]['值'])
tmp=tmp.reset_index()
del tmp['值']
del tmp['index']

temp2=SquareTable.from_data(tmp)

temp2.summary()
#可求边缘分布概率
row, col = temp2.marginal_probabilities
#方形列联表检验
temp2.symmetry()

#程序中并没有比例差和置信区间的估计方法.看来需要手动去求.或者对statsmodels进行更深入探索.

#(2)kappa
data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/癌症与诊断.csv")
temp=np.array([[22,2,2,0],[5,7,14,0],[0,2,36,0],[0,1,17,10]])
#与书中值一致.
cohens_kappa(temp)
#2.模型
#(1)配对边缘logistic。
#这个模型非常怪,并没有属于任何已知常用的模型,需要自己手动去拟合logit函数.
#由于只存在两个点直接调用logit函数能得到近似的结果,不用加入似然过程.
#-0.78236221   -   -0.88589346 =0.10353125
#关于边缘的模型可以使用GEE。只是GEE进行拟合检验需要额外的假设.



#(2)条件模型 叫做ConditionalLogit。但是仅适用于
#名义的,二项的多分类的和poisson
#对于有序的支持不够.

data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/环保.csv")
tmp=pd.DataFrame()
zhi=0
# print(data)

for i in range(0,4):
    data_temp=data.loc[i]
    for j in range(0,data_temp['值']):
        zhi+=1
        temp = {'是': 1, "否": 0}
        for weizhi,f in enumerate(['降低生活水平','付更高的税']):
            temp2 = [0, 0, 0]
            temp2[0]=zhi
            temp2[1]=temp[data_temp[f]]
            temp2[2]=weizhi
            tmp=tmp.append(pd.DataFrame([{'记录':temp2[0],
                                     'y':temp2[1],
                                     "x":temp2[2]}]))

tmp=tmp.reset_index()
del tmp['index']
# tmp.to_csv(r"D:/书籍资料整理/属性数据分析/环保_展开.csv")
x=np.array(tmp['x'])
xx=x[:, None]
yy=np.array(tmp['y'])
g=np.array(tmp['记录'])

model=ConditionalLogit(yy,xx, groups=g)
#这里fit的时候会提示删除了905个没有组内方差的.也就是组内数据相同的值.

result=model.fit()
result.summary()
#书中给出的结果为23%,由于书中直接用n求得.
#而用软件为21%。
#R与这个软件一直所以不研究了
# (3)对称logistic  没有找到实现方式并且网上关于这个的文章很少.



