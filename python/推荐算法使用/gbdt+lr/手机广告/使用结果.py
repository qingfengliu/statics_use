import pandas as pd
import numpy as np
import joblib
import time
import configparser
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder

pd.set_option('max_row',200)
pd.set_option('max_columns',100)

cf=configparser.ConfigParser()
filename=cf.read('')
base_dir=cf.get('path','data_path')
norm=joblib.load(base_dir+'/归一化.pkl')
types=joblib.load(base_dir+'/')

model1=joblib.load(base_dir+'/')  #lr
model2=joblib.load(base_dir+'/') #gbdt

def data_make_prodata(data):
    data['userid']=data['userid'].astype(str)
    data['sex'].fillna('未知',inplace=True)
    data['position'].fillna(0, inplace=True)
    data['types'].fillna('未知', inplace=True)
    data['age'].fillna('未知', inplace=True)
    data['itemnum'].fillna(0, inplace=True)
    data['1']

    # item是物品ID
    data_spare=pd.DataFrame(item.categories_).T
    data_spare['1']='1'
    data_spare.columns=['splancode','1']
    data=data.merge(data_spare,how='inner',on='1') #其他信息与itemid做笛卡尔积用户预测配个物品评分
    data=data[['...']]
    return data

def data_transform_gbdt(data):
    #这里一堆one_hot组合成送入gbdt的样子
    df1=types.transform(data['types'].values.reshape(-1,1)).toarray()
    data=pd.concat([data,pd.DataFrame(df1.astype(np.float16),columns=[sex:+x for x in sex_apply.categories_[0]])],axis=1)
    return data

def goto_gbdt(data):
    train_new_feature=model2.apply(data)
    train_new_feature=train_new_feature.reshape(-1,20)
    #enc也是一个pkl文件
    temp=enc.transform(train_new_feature).toarray()
    train_new_featyre2=np.array(temp,dtype=np.float16)

    train_new_feature3=pd.DataFrame(train_new_feature2,columns=['gbdt:'+str(x) for x in range(0,train_new_feature2.shape[1])])
    data_use=pd.concat([data,train_new_feature3],axis=1)
    return data_use

data_result=pd.DataFrame()
data=pd.read_csv(base_dir+'',chunksize=50000) #放置内存存不下
time_start=time.time()
for data_all in data:
    data_all=data_make_prodata(data_all)
    data_all=data_transform_gbdt(data_all)
    data_splancode=data_all[['user_id','splancode']].copy() #用户ID没有计入x
    data_use=data_all[[x for x in data_all.columns if x not in ['...']]]
    results=model1.predict_proba(data_use)
    data_splancode['score']=results[:,0]
time_end=time.time()
sum_t=(time_end-time_start)
print('time cost',sum_t,'s')

print(data_result.shape)

data_result=data_result.groupby(['member_id','splancode','name']).head(1)
data_result.head()
data_result.to_csv(base_dir+'/LR推荐结果.csv',index=True)
data_result['splancode'].value_counts()