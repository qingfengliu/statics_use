#xgboost 与随机森林进行比较，采用10这交叉验证,
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve,mean_squared_error
from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score,roc_auc_score

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# print(mnist.keys())
X, y = mnist["data"], mnist["target"]
# print(type(X))
# print(X.shape)
# print(y.shape)
y = y.astype(np.uint8)

some_digit = X[0]
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

train = xgb.DMatrix(X_train, label=y_train)

test = xgb.DMatrix(X_test, label=y_test)
params={
    'booster':'gbtree',
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'learning_rate':0.1,
    'gamma':0.1,
    'subsample':0.8,
    'max_depth':5,
    'reg_alpha':50,
    'colsample_bytree':0.7,
    'min_child_weight':200,
    'seed':22
}

model_r = xgb.XGBRegressor(booster='gbtree',objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.1,
    gamma=0.1,
    subsample=0.8,
    max_depth=5,
    reg_alpha=50,
    colsample_bytree=0.7,
    min_child_weight=200,
    seed=22)

# model_r.fit(X_train, y_train)
#,callbacks=[xgb.callback.print_evaluation(show_stdv=True)]

print('正在跑 测试交叉验证 2')
y_train_score = cross_val_predict(model_r, X_train, y_train_5, cv=10)   #每个数据预测值
y_pred_train = (y_train_score >= 0.5)
print('训练集准确率:',accuracy_score(y_train_5, y_pred_train))
print('训练集精度:',precision_score(y_train_5, y_pred_train))
print('训练集召回率:',recall_score(y_train_5, y_pred_train))

print('混淆矩阵:',confusion_matrix(y_train_5, y_pred_train))
print('auc:',roc_auc_score(y_train_5, y_pred_train))