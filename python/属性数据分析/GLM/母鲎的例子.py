import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#1.单变量的logistic模型
data=pd.read_excel(r"D:/书籍资料整理/属性数据分析/母鲎及其追随者案例.xlsx")
data['追随者数']=data['追随者数'].apply(lambda x:int(bool(x)))
# print(data)

model=smf.glm(formula="追随者数~甲壳宽度",data=data,family=sm.families.Binomial())
results = model.fit()
#参数展示的结果是wald置信区间
#书中提到的似然比检验由参数为0时LL-Null似然函数值/
#全模式(当前模式)的对数似然函数值Log-Likelihood给出
print(results.summary())
#预测
tmp=pd.DataFrame(data.loc[0]).T   #单行这样建立Dataframe会转置要转置回来
tmp.iloc[0,2]=26.5

results.predict(tmp)
#协方差矩阵用于求预测的置信区间.
results.cov_params()

#2.多元logistic模型
data=pd.read_excel(r"D:/书籍资料整理/属性数据分析/母鲎及其追随者案例.xlsx")
data['追随者数']=data['追随者数'].apply(lambda x:int(bool(x)))
model=smf.glm(formula="追随者数~甲壳宽度+C(颜色)",data=data,family=sm.families.Binomial())
results = model.fit()
#这个结果与书中的稍有不同,因为书中将参数全为0表示了"深"。而这里参数全为0表示"稍浅".
results.summary2()

#检验,模型是否合适,采用的是似然比检验。将这个模型的似然数与去掉颜色的模型的似然数比较
#前边已经求出了去掉的.其统计量为7。服从df=3的卡方分布,自由度为两模型变量的差值.
#(颜色为4个档引入了3个变量)
#p=0.07有些轻微的证据,也算较小了
#并且AIC BIC都变小了.

#3.交互效应
data=pd.read_excel(r"D:/书籍资料整理/属性数据分析/母鲎及其追随者案例.xlsx")
data['追随者数']=data['追随者数'].apply(lambda x:int(bool(x)))
model=smf.glm(formula="追随者数~甲壳宽度+C(颜色)+甲壳宽度:C(颜色)",data=data,family=sm.families.Binomial())
results = model.fit()
results.summary2()
