#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

data=pd.read_csv(r'D:\书籍资料整理\kaggle\titanic\train.csv')
#注释一下列名
#survival	是否存活	0 = No, 1 = Yes
#pclass	票类型	1 = 1st, 2 = 2nd, 3 = 3rd
#sex	性别
#Age	年龄
#sibsp	泰坦尼克号上的兄弟姐妹/配偶
#parch	# 泰坦尼克号上的父母/孩子
#ticket	Ticket number	票号
#fare	票价
#cabin	房间号
#embarked	出发港	C =  瑟堡, Q = 昆士城, S = 南安普敦

data=data[['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']] #去掉可能的无关值

#这里name,sexm,ticket,cabin,embarked为字符串数据
#首先看下空值率
print(float(data['Pclass'].notnull().count()/data.shape[0]))
print(float(data['Sex'].notnull().count()/data.shape[0]))
print(float(data['Age'].notnull().count()/data.shape[0]))
print(float(data['SibSp'].notnull().count()/data.shape[0]))
print(float(data['Parch'].notnull().count()/data.shape[0]))
print(float(data['Fare'].notnull().count()/data.shape[0]))
# print(float(data['Cabin'].notnull().count()/data.shape[0]))
print(float(data['Embarked'].notnull().count()/data.shape[0]))

sex=LabelEncoder()
sex.fit(data['Sex'])
data['Sex']=sex.transform(data['Sex'])

embarked=LabelEncoder()
embarked.fit(data['Embarked'])
data['Embarked']=embarked.transform(data['Embarked'])

joblib.dump(sex,r'D:\书籍资料整理\kaggle\titanic\sex.pkl')
joblib.dump(sex,r'D:\书籍资料整理\kaggle\titanic\embarked.pkl')

X=data[[x for x in data.columns if x not in ['Survived','PassengerId']]]
y=data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

# train = xgb.DMatrix(X_train, label=y_train)
# test = xgb.DMatrix(X_val, label=y_val)
xgb_reg = xgb.XGBClassifier()
xgb_reg.fit(X_train, y_train)


y_pred = xgb_reg.predict(X_val)
y_score = xgb_reg.predict_proba(X_val)


from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score,roc_auc_score,mean_squared_error
val_error = mean_squared_error(y_val, y_pred)  # Not shown
print("Validation MSE:", val_error)  # Not shown

print('测试集准确率:',accuracy_score(y_val, y_pred))
print('测试集精度:',precision_score(y_val, y_pred))
print('测试集召回率:',recall_score(y_val, y_pred))
print('auc:',roc_auc_score(y_val, y_score[:,1]))

data_test=pd.read_csv(r'D:\书籍资料整理\kaggle\titanic\test.csv')
data_test=data_test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']] #去掉可能的无关值
data_test['Sex']=sex.transform(data_test['Sex'])
data_test['Embarked']=embarked.transform(data_test['Embarked'])
data_test.head()

#Pclass	Sex	Age	SibSp	Parch	Fare	Embarked
X_test=data_test[[x for x in data_test.columns if x not in ['PassengerId']]]

y_test_pre=xgb_reg.predict(X_test)
data_test['Survived']=y_test_pre
data_test.head()

data_test=data_test[['PassengerId','Survived']]


data_test.to_csv(r'D:\书籍资料整理\kaggle\titanic\output.csv',index=False)


#差不多是这个logloss
from sklearn.metrics import log_loss
log_loss(y_val, y_pred)

