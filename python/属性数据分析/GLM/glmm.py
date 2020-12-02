import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
                                                PoissonBayesMixedGLM)
import statsmodels.api as sm
import statsmodels.formula.api as smf
#对于python还无法模拟有序的和多分类。
#书中说二分结果可以直接推广是否意味着没有类别-基类的那种模型,只能通过类别-其他来推广
#python目前已知无法针对累计GLMM

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

tmp['诊断严重程度']=tmp['诊断严重程度'].replace({'轻微':0,'严重':1})
tmp['治疗']=tmp['治疗'].replace({'标准':0,'新药':1})
tmp=tmp.reset_index()
del tmp['index']
tmp=tmp.rename(columns={'周数':'zhous','值':'zhi','组':'zu','诊断严重程度':'severity','治疗':'drug'})
tmp.to_csv(r"D:/书籍资料整理/属性数据分析/抑郁症治疗_展开.csv")
random = {"a": '0 + C(zu)'}

model = BinomialBayesMixedGLM.from_formula(
    'zhi ~ severity + drug + zhous+drug:zhous', random, tmp)
result = model.fit_vb()
#给出的结果大致上与书中的结果差不多,估计差异在
#书中给出的结果为使用高斯-埃尔米特求积
#而statsmodels使用的是贝叶斯方法.
#结果给出的是方差可能要开根号才能求标准差
#另外
print(result.summary())

data=pd.read_csv(r"D:/书籍资料整理/属性数据分析/老鼠.csv")
random = {"a": '0 + C(簇)'}

model = BinomialBayesMixedGLM.from_formula(
    '死亡 ~ C(组) ', random, data)
result = model.fit_vb()
result.summary()