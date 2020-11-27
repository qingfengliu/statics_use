import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/抑郁症治疗.csv")
tmp=pd.DataFrame()
zu=0
for i in range(0,4):
    data_temp=data.loc[i]
    for key in ['NNN','NNA','NAN','NAA','ANN','ANA','AAN','AAA']:
        renshu=data_temp[key]
        for j in range(0, renshu):
            zu+=1
            for shij in range(0,3):
                if key[shij]=='N':
                    zhi=1
                else:
                    zhi=0
                temp = pd.DataFrame([{'周数': shij , '值': zhi,'组':zu}])
                temp['诊断严重程度'] = data_temp['诊断严重程度']
                temp['治疗'] = data_temp['治疗']
                tmp=tmp.append(temp)
# tmp.to_csv('D:/结果数据_抑郁症治疗.csv',encoding='gbk')
tmp['诊断严重程度']=tmp['诊断严重程度'].replace({'轻微':0,'严重':1})
tmp['治疗']=tmp['治疗'].replace({'标准':0,'新药':1})
tmp=tmp.reset_index()
del tmp['index']
va = sm.cov_struct.Autoregressive()
fam = sm.families.Binomial()
ind = sm.cov_struct.Independence()
#与书中结果一致
mod = smf.gee("值 ~ 诊断严重程度 + 治疗 + 周数+治疗:周数", "组", tmp, cov_struct=ind,family=fam)
res = mod.fit()
res.summary()

#2.多元GEE 疑似NominalGEE
#3.有序 OrdinalGEE


