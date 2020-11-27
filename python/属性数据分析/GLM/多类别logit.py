import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pandas.api.types import CategoricalDtype


data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/钝吻鳄例子.csv")

#1.基线-类别logit
data['食物选择']=data['食物选择'].replace({'I':3,'F':2,'O':1})
mdl = smf.mnlogit(formula='食物选择~长度',data=data)
result = mdl.fit()
#与书中不同的还是去第一个数值为基线类,由于因变量是类别
#与顺序无关所以这里取O=1,得到与书中一样的结果.
#得到两个方程是无脊椎-其他
#鱼-其他
#若研究无脊椎与鱼的关系则使用(1)-(2)得到
result.summary()

#2.有序响应变量的累积logit模型
data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/政治意识与党派.csv")
# data['意识形态']=data['意识形态'].replace({'很自由':1,'有点自由':2,'中等':3,'有点保守':4,'很保守':5})
data['政治党派']=data['政治党派'].replace({'民主党人':1,'共和党人':0})
tmp=pd.DataFrame()
for i in range(0,20):
    tmp=tmp.append([data.loc[i]]*data.iloc[i]['值'])
tmp=tmp.reset_index()
del tmp['值']
del tmp['index']
# tmp.to_csv(r'D:/书籍资料整理/属性数据分析/政治意识与党派_整理数据.csv')
#得到的结果显示,自变量参数是反的.这个可以解释,因为使用的是α-βx展示
#书中的结果是α+βx
#但是截距从第二个开始就相去甚远很难找到解释理由,OrderedModel这个功能
#并非包内本身带的,文档也几乎没有提到.
#这个是将要被statsmodels带入的功能并没有完善待后续.
tmp['意识形态'] = tmp['意识形态'].astype('category')
s = pd.Series(["a", "b", "c", "a","d","e"])
cat_type = CategoricalDtype(categories=['很自由','有点自由','中等','有点保守','很保守'],ordered=True)#categories必须是一个列表
tmp['意识形态']=tmp['意识形态'].astype(cat_type)

modf_logit = OrderedModel.from_formula("意识形态~政治党派", tmp,distr='logit')
resf_logit = modf_logit.fit(method='bfgs')
print(resf_logit.summary())


data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/心灵伤害与SES.csv")

data['心理伤害']=data['心理伤害'].replace({'健康':0,'轻度':1,'中等':2,'受损':3})
modf_logit = OrderedModel.from_formula("心理伤害~SES+生活事件", data,distr='logit')
resf_logit = modf_logit.fit()
resf_logit.summary()