from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
def checkVIF_new(df):


    name = df.columns
    x = df.values
    VIF_list = [variance_inflation_factor(x,i) for i in range(x.shape[1])]
    VIF = pd.DataFrame({'feature':name,"VIF":VIF_list})
    return VIF

data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/表6-1.csv")
data2=pd.DataFrame(data[['x1','x2','x3','x4','x5']])
data2['c']=1

vif=checkVIF_new(data2)
print(vif)