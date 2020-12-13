import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import t
import dwtest
from scipy import stats
#一、异方差问题
data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/表4-1.csv")
#异方差判断
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

model=smf.ols('财政收入~地区生产总值',data=data)
result=model.fit()

result.summary()
res=data
res['residual']=result.resid

#求spearman相关系数手动,做spearman相关检验
res[['地区生产总值','residual']].corr(method='spearman')
t1=(31-2)** 0.5*-0.247984/(1+0.247984** 2)** 0.5
t.cdf(t1,df=29)

#可以从残差图看出明显的异方差

# plt.scatter(res['地区生产总值'], res['residual'])
# plt.show()
#1.加权最小二乘法
#加权最小二乘法,需要构建一个权重。
#python中也无法自动寻找一个合适的m
#所以,只能通过找似然值最小的,作为合适的.
#一般从-2-2试。每隔0.5取一个
#直接取书中的结果2,但是wls内部会自动取倒数。
data['w']=data['地区生产总值'].apply(lambda x:x**-2)

model=smf.wls('财政收入~地区生产总值',data=data,weights=data['w'])
result=model.fit()

result.summary()


res=data
res['residual']=result.resid*(model.weights**0.5)
#做加权残差图
# plt.scatter(res['地区生产总值'], res['residual'])
# plt.show()
#2.BOX-BOX变换
data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/表4-3.csv")

#使用lmbda=None,得出与书中描述不符.所以只能指定lmbda
x_norm = stats.boxcox(data['财政收入'],lmbda=0)
data['y']=x_norm
x_norm = stats.boxcox(data['地区生产总值'],lmbda=0)
data['x']=x_norm
model=smf.ols('y~地区生产总值',data=data)
result=model.fit()
result.summary()

#如果要得到残差,那么需要将式子转换会普通的线性回归.
#然后计算预测值得到残差
#这里要说到。延展的结论书中只提到而没作为试验
#1.lmbda=0就是log变换了.
#如果仅变换Y或者x就是换一个模型了,如果残差合适,说明线性模型不适合
#如果同时对x,y做变换效果好的话就说明消除了异方差性
#本例就需要对x和y同时做,消除的异方差性.
#并且残差图可以不对方程做变换回来的操作可直接看。
#下边的代码就做了这个试验
# x_norm = stats.boxcox(data['财政收入'],lmbda=0)
# data['y']=x_norm
# x_norm = stats.boxcox(data['地区生产总值'],lmbda=0)
# data['x']=x_norm
# model=smf.ols('y~x',data=data)
# result=model.fit()
# result.summary()
# res=data
# res['residual']=result.resid
#做加权残差图
# plt.scatter(res['x'], res['residual'])
# plt.show()
#二、残差自相关
data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/第二章-一元回归分析.csv")
model=smf.ols('人均支出~人均收入',data=data)
result=model.fit()
res=data
#从统计量结果看残差有自相关性.
res['residual']=result.resid
#做PW检验
statistic, pvalue =dwtest.dwtest('人均支出~人均收入',data=data)
statistic, pvalue
#1.迭代法(Cochrane-Orcutt)
#2.差分法(广义差分法)
#用差分法求相关系数(durbin两步法)
data['tsy1']=data['人均支出'].diff(1)  #进行一次差分
data['tsx1']=data['人均收入'].diff(1)  #进行一次差分
# data['ts1'].dropna()

model=smf.ols('tsy1~tsx1-1',data=data)
result=model.fit()
#使用差分法与书中一样了
result.summary()
#迭代法和差分法本质是一样的,由于python没有封装好的函数
#那么就只做下差分法的试验了

#三、异常值与强影响
data=pd.read_csv(r"D:/书籍资料整理/应用回归分析/表3-7.csv")

model=smf.ols('y~x1+x2',data=data)
result=model.fit()
infl = result.get_influence()
summ_df = infl.summary_frame()
# summ_df.to_csv(r"D:/书籍资料整理/temp.csv")
#该结果给出了:
#库克距离,cook_d
#学生化残差:standard_resid
#杠杆值hi:hat_diag
#删除学生化残差:student_resid
#库克距离和杠杆值可供判断y异常
#学生化残差和删除学生化残差可供判断x异常
summ_df