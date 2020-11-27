import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
data=pd.read_excel(r"D:/书籍资料整理/属性数据分析/母鲎及其追随者案例.xlsx")

# possion回归 广义线性模型.
#statsmodels.formula.api这个包里可以让你使用字符串语法建立模型。
#formula 代表模型包含变量，和变量类型。如果用C(变量),则变量是属性变量
#family表示广义线性模型支持的分布族
model=smf.glm(formula="追随者数~甲壳宽度",data=data,family=sm.families.Poisson())
results = model.fit()
results.summary()

