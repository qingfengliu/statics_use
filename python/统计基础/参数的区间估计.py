#1.均值的估计
#(1)大样本,z检验

import pandas as pd
from scipy.stats import norm
import math
data=pd.read_csv(r'D:\书籍资料整理\统计学\example5_1.csv',encoding='gbk')

#python无法直接求z检验的置信区间只能借助自己计算
norm.interval(0.90,loc=data['耗油量'].mean(),scale=data['耗油量'].std()/math.sqrt(data['耗油量'].count()))
#那么同理其他的置信区间可通过,直接调用scipy包里的相应分布函数


