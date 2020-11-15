import pandas as pd
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
#单变量的差异检验
data=pd.read_csv(r'D:\书籍资料整理\统计学\example7_1.csv',encoding='gbk')
chisquare(data['人数'])

#单变量与期望差异
data=pd.read_csv(r'D:\书籍资料整理\统计学\example7_2.csv',encoding='gbk')
chisquare(data['离婚家庭数'],data['期望'])

#列联表的卡方检验(独立性检验)
#数据一开始并不是一个列联表,需要处理一下.
data=pd.read_csv(r'D:\书籍资料整理\统计学\example7_3.csv',encoding='gbk')
data=pd.crosstab(data['满意度'],data['地区'])
chi2_contingency(data)
#返回值第一个是卡方统计量,第二个是P值.
print(data)

#列联表相关系数暂时没有找到方法计算,先略过