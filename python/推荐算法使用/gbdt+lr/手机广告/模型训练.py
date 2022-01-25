import pandas as pd
import numpy as np
import joblib
import time
import configparser
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

pd.set_option('max_row',200)
pd.set_option('max_columns',200)

cf=configparser.ConfigParser()
filename=cf.read('model.ini')
base_dir=cf.get("path","data_path")

data_all=pd.read_csv(base_dir+'/数据处理.csv')

data_all['y']=data_all['if_click']

data_all=data_all[[x for x in data_all.columns if x not in ['if_click']]]

data_use=data_all[[x for x in data_all.columns if x not in ['member_id','client_time']]]

X=data_use[[x for x in data_all.columns if x not in ['y']]]

Y=data_use['y']

gbml=GradientBoostingClassifier(n_estimators=20,random_state=10,sumsample=0.6,max_depth=7,min_samples_split=900)
gbml.fit(X,Y)
train_new_feature=gbml.apply(X)
train_new_feature=train_new_feature.reshape(-1,20)
enc=OneHotEncoder()
enc.fit(train_new_feature)

train_new_feature2=np.array(enc.transform(train_new_feature).toarray(),dtype=np.float16)

train_new_feature3=pd.DataFrame(train_new_feature2,columns=['gbdt:'+str(x) for x in range(0,train_new_feature2.shape[1])])
data_all=pd.concat([data_all,train_new_feature3],axis=1)

data_use=data_all[[x for x in data_all.columns if x not in ['member_id','client_time']]]

oversample=data_use[data_use['y']==1].sample(frac=10,replace=True)
print(oversample['y'].value_counts())

oversample2=data_use[data_use['y']!=1].sample(frac=0.2)

data_use=pd.concat([oversample,oversample2],axis=0)

x_train,x_test=train_test_split(data_use,train_size=0.8)

X=x_train[[x for x in x_train.columns if x not in ['y']]]
Y=x_train['y']

X_2=x_test[[x for x in x_test.columns if x not in ['y']]]
Y_2=x_test['y']

time_start=time.time()
model1=LogisticRegression(max_iter=5000,class_weight='balanced')
model.fit(X,Y)
time_end=time.time()

sum_t=(time_end-time_start)
print('time cost',sum_t,'s')

Y_PRE2=model1.predict_proba(X_2)
Y_pre=model1.predict(X_2)

print(metrics.f1_score(Y_2,Y_pre))
print(metrics.classification_report(Y_2,Y_pre))
print('模型AUC:',metrics.roc_auc_score(Y_2,Y_PRE2[:,1]))

joblib.dump(model1,base_dir+'/数据处理过程保存/lr_splan_code_2.pkl')
joblib.dump(gbml,base_dir+'/数据处理过程保存/gbdt_splan_code.pkl')
joblib.dump(enc,base_dir+'/数据处理过程保存/gbdt_one_hot_splan_code.pkl')