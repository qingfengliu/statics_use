from scipy import stats
import pandas as pd
import rpy2.robjects as robjects
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#1.单因素方差分析

data=pd.read_csv(r'D:\书籍资料整理\统计学\example8_1.csv',encoding='gbk')
# data=data.melt().groupby('variable').sum()
# data.reset_index(inplace=True)   #索引转换成列
stats.f_oneway(data.loc[:,'品种1'],data.loc[:,'品种2'],data.loc[:,'品种3'])
#结果仅给出了P值并没有,网上相关文档提到模型参数估计所以这里 ,给出调用R.目前使用不成功套件需要安装windows编译工具


robjects.r(
    '''
example8_2<-read.csv("D:/书籍资料整理/统计学/example8_2.csv")
attach(example8_2)
model_1w<-aov(chanliang~pinzhong)
    '''
)
#获得F检验值
robjects.r('summary(model_1w)')
#获得方差分析参数
robjects.r('model_1w$coefficients')
#获得效应量
robjects.r('library(DescTools)')
robjects.r('EtaSq(model_1w,anova = T)')

#LSD,这里会给出,书中详细信息中的信息.
data=pd.read_csv(r'D:\书籍资料整理\统计学\example8_1.csv',encoding='gbk')
data=data.melt()
# data.reset_index(inplace=True)   #索引转换成列

res = ols("value ~ C(variable)", data).fit()
#付上使用statsmodels包来分析得到方差分析结果
anova_lm(res)
pw = res.t_test_pairwise("C(variable)")
#LSD多重比较
pw.result_frame

#HSD
pairwise_tukeyhsd(data['value'],data['variable'])

#2.(1)多因素方差分析
data=pd.read_csv(r'D:\书籍资料整理\统计学\table8_4.csv',encoding='gbk')
data=data.melt(id_vars=['施肥方式'],value_vars=['品种1','品种2','品种3'],var_name='品种')
#stats没有多因素方差分析的,所以统一使用statsmodels包
res1 = ols("value ~ C(施肥方式)+C(品种)", data).fit()
anova_lm(res1)
#若求拟合参数还需使用R
robjects.r(
    '''
example8_5<-read.csv("D:/书籍资料整理/统计学/example8_5.csv")
attach(example8_5)
model_2wm<-aov(产量~品种+施肥方式)
    '''
)
#获得F检验值，因为编码问题汉字显示会有问题,并且会串行,就结果说是没问题的
robjects.r('summary(model_2wm)')
#获得方差分析参数
robjects.r('model_2wm$coefficients')
#获得效应量
robjects.r('library(DescTools)')
robjects.r('EtaSq(model_2wm,anova = T)')

#2.(2)考虑交互效应
# smf:最小二乘法,构建线性回归模型 。。
res2 = ols('value ~ C(施肥方式) + C(品种) + C(施肥方式)*C(品种)', data).fit()
# anova_lm:多因素方差分析
anova_lm(res2)
#若求拟合参数还需使用R
robjects.r(
    '''
example8_5<-read.csv("D:/书籍资料整理/统计学/example8_5.csv")
attach(example8_5)
model_2wm<-aov(产量~品种+施肥方式+品种:施肥方式)
    '''
)
#获得F检验值，因为编码问题汉字显示会有问题,并且会串行,就结果说是没问题的
robjects.r('summary(model_2wm)')
#获得方差分析参数
robjects.r('model_2wm$coefficients')
#获得效应量
robjects.r('library(DescTools)')
robjects.r('EtaSq(model_2wm,anova = T)')

#(3)两模型比较。结果P值为0.37说明交互效应不明显.
#anova用于模型比较时,一个模型必须包含在另一个模型中.
anova_lm(res1,res2)