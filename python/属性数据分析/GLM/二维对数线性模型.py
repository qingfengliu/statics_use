import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/打鼾与心脏病.csv")
tmp=pd.DataFrame()
for i in range(0,8):
    tmp=tmp.append([data.loc[i]]*data.iloc[i]['值'])

tmp['心脏病']=tmp['心脏病'].replace({'是':1,'否':0})
tmp['打鼾']=tmp['打鼾'].replace({'从不':0,'偶尔':2,'几乎每晚':4,'每晚':5})
tmp=tmp.reset_index()
del tmp['值']
del tmp['index']
# print(tmp.shape)

model=smf.glm(formula="心脏病~打鼾",data=tmp,family=sm.families.Binomial())
results = model.fit()
results.summary2()

#附赠,probit回归
# model=smf.probit(formula="心脏病~打鼾",data=tmp)
# results =model.fit()
# print(results.summary())