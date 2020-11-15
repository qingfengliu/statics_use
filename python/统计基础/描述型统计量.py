import pandas as pd

data=pd.read_csv(r'D:\书籍资料整理\统计学\example3_1.csv',encoding='gbk')
data['分数'].mean()   #均值
data['分数'].median()   #中位数
data['分数'].mode()    #众数
data['分数'].max()-data['分数'].min()  #极差
data['分数'].quantile(0.75)-data['分数'].quantile(0.25)   #四分位差
data['分数'].var()      #方差
data['分数'].std()      #标准差
data['分数'].mean()/data['分数'].std()    #变异系数
#标准分数-z分数没有标准函数,需要自己编写

data['分数'].skew()    #偏度计算
data['分数'].kurt()   #峰度计算

print(data.describe())  