import pandas as pd
data=pd.read_csv(r'D:\书籍资料整理\时间序列分析_王燕\file22.csv',index_col='month')

data.index=pd.to_datetime(data.index,format='%Y/%m/%d')

#1.使用水平模型,提取均值序列中蕴含的相关信息
#2.检验残差序列是否具有条件异方差特性
#3.