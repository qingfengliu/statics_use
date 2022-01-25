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
filename=cf.read('model.ini')
base_dir=cf.get("path","data_path")
print(base_dir)

data_all=pd.read_csv(base_dir+'/数据/train.csv')
print(data_all.shape)
print(data_all.head())
exit()
data_all['sex'].fillna('未知',inplace=True)
data_all['position'].fillna('未知',inplace=True)

data_all=data_all[['.....']]

pd.set_option('max_row',200)
pd.set_option('max_columns',100)

types=OneHotEncoder()
types.fit(data_all['types'].values.reshape(-1,1))
#types.categories_

df1=types.transform(data_all['type'].values.reshape(-1,1)).toarray()
data_all=pd.concat([data_all,pd.DataFrame(df1,columns=['type:'+x for x in types.categories_[0]])],axis=1)

data_all=data_all[[x for x in data_all.columns if x not in ['types']]]

data_use=data_all[['..连续性变量..']]

norm=MinMaxScaler()
norm.fit(data_use)
data_samilar_s=pd.DataFrame(norm.transform(data_use),columns=data_use.columns)

#将0，1变量与归一化的连续性变量合并到一起
data_all=pd.concat([data_all[[x for x in data_all.columns if x not in ['..连续性变量..']]],data_samilar_s])

data_all.to_csv(base_dir+'/数据处理.csv')

joblib.dump(norm,base_dir+'/归一化.pkl')
joblib.dump(norm,types+'/types_onehot.pkl')
