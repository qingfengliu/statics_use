import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy import stats
from statsmodels.stats.multivariate import test_cov_oneway
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
import statsmodels.stats.oneway as smo
data=pd.read_csv(r"D:\书籍资料整理\多元统计分析\例2-1.csv")

#1.正态性检验
for key,value in data.iteritems():
    if key in ['省份','城市']:
        continue
    key+':statistic %s,pvalue:%s' %(shapiro(value)[0],shapiro(value)[1])
    #仅有人均地区生产总值和公共财政支出满足正态
#K-S不适合小样本,参见统计基础-假设检验

#2.多变量检验表证明省份对Y有影响
mod = _MultivariateOLS.from_formula('人均地区生产总值+公共财政支出 ~ 省份',data)
result=mod.fit(method='svd')
result.mv_test()
#3.多元统计-协方差阵检验


temp_data=[]
temp_name=[]
for name, group in data[['省份','人均地区生产总值','公共财政支出']].groupby(['省份']):
    temp_data.append(np.cov(np.asarray(group[['人均地区生产总值','公共财政支出']].T)))
    temp_name.append(name)


#statistic_base 是Box's M统计量
#pvalue是书中给出的p值
test_cov_oneway(temp_data,[5,5,5])

#4.误差方差分析
temp_data=[]
temp_name=[]
for name, group in data[['省份','人均地区生产总值','公共财政支出']].groupby(['省份']):
    temp_data.append(np.array((group['人均地区生产总值']-group['人均地区生产总值'].mean())))
    # temp_name.append(name)
temp_data=np.array(temp_data)
res0 = smo.test_scale_oneway(temp_data, method='equal', center='mean',
                             transform='abs', trim_frac_mean=0.2)

'statistic:%s,df2:%s,Sig.:%s,pvalue:%s' %(res0.statistic,res0.df_num,res0.df_denom,res0.pvalue)


