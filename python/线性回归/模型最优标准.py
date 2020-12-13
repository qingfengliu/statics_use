import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

#一、异方差问题
data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/表5-1.csv")
#异方差判断

model=smf.ols('y~x1',data=data)
result=model.fit()
#提供了决定系数、调整后的决定系数(越大越好)
#提供了AIC、BIC(越小越好)
result.summary()
