import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer,calculate_bartlett_sphericity,calculate_kmo

data=pd.read_csv(r"D:\书籍资料整理\多元统计分析\例5-3.csv")
x=data[['企业单位数','流动资产','资产总额','负债总额','主营业务','利润总额','销售利润率']]

#SPSS 使用了principal
#网上的一个解释 其中第一主成分对初始变量集的方差解释性最大，
#随后的每一个主成分都最大化它对方差的解释程度，同时与之前所有的主成分都正交
#然后方差旋转使用正交旋转,各个因子彼此独立。表现为载荷矩阵中的元素更倾向于与0和±1
#
fa = FactorAnalyzer(n_factors=2,rotation='varimax',method='principal')
fa.fit(x)
#原始相关性矩阵
fa.corr_

#检验,相关矩阵是否是单位矩阵
calculate_bartlett_sphericity(x)
#KMO检验,总得KMO应该大于0.6
calculate_kmo(x)
#公因子方差
fa.get_communalities()
#载荷矩阵
fa.loadings_

#因子贡献率
#variance 因素方差,proportional_variance 比例因子方差,cumulative_variances 累计比例因子方差
fa.get_factor_variance()

#没有提供得分矩阵,得分矩阵一般用于将原始数据转换为因子分析后的数据.可以使用transform函数

#斜交例子
model = FactorAnalyzer(n_factors=2,rotation='promax',method='principal')
model.fit(x)
#pattern matrix
model.loadings_
#应该是文档出现错误 成分相关矩阵 应该是phi_ 不是psi
#correlation matrix
model.phi_

#载荷矩阵=pattern matrix*correlation matrix