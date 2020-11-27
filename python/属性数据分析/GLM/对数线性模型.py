import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
#1.关于建模
data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/来世.csv")
#对数线性模型,适用于glm的泊松回归。因变量是单元格中的期望值,分组后的数据.
#得到一样的结果
model=smf.glm(formula="值~C(种族)+C(来世)",data=data,family=sm.families.Poisson())
results = model.fit()
results.summary()

#交叉模型报错无法建模因为,无交互效应?

data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/酒香烟大麻.csv")
#对数线性模型,适用于glm的泊松回归。因变量是单元格中的期望值,分组后的数据.
#得到一样的结果

data['酒']=data['酒'].replace({'是':1,'否':0})
data['香烟']=data['香烟'].replace({'是':1,'否':0})
data['大麻']=data['大麻'].replace({'是':1,'否':0})
model=smf.glm(formula="值~C(酒)+C(香烟)+C(大麻)",data=data,family=sm.families.Poisson())
results = model.fit()
results.summary()
tmp=pd.DataFrame(data.loc[0]).T   #单行这样建立Dataframe会转置要转置回来
del tmp['值']
#与书中值一致
results.predict(tmp)

#交互模型
model=smf.glm(formula="值~C(酒):C(大麻)+C(香烟):C(大麻)",data=data,family=sm.families.Poisson())
results = model.fit()
results.summary()
tmp=pd.DataFrame(data.loc[0]).T   #单行这样建立Dataframe会转置要转置回来
del tmp['值']
#与书中值一致
results.predict(tmp)

#2.检验
#(1)拟合检验
# 关于检验summary给出的值与书中一样,因为他们取自的都是
#分组表,Deviance与G2一致,Pearson chi2与X2一致,Df Residuals与DF一致
#就是没给P值,需要自己拟合
#(2)拟合点检验
infl = results.get_influence(observed=False)
summ_df = infl.summary_frame()
#接口提供了更专业的图检验
# fig = infl.plot_index(y_var='cooks', threshold=2 * infl.cooks_distance[0].mean())
# fig.tight_layout(pad=1.0)
# plt.show(block=True)

