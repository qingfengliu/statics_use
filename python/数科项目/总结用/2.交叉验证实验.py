import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve,mean_squared_error
from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score,roc_auc_score
from sklearn.datasets import fetch_openml


#绘制精度召回率 阈值图
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    # plt.axis([-50000, 50000, 0, 1])             # 这个代码用于指定坐标轴宽度

#绘制ROC曲线
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown

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
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_clf.fit(X_train, y_train_5)  #这里暂时使用随机森林来拟合,
#交叉验证主要是验证训练集参数的很有效指标, 对于测试集这样的处理是否有用呢？
print('正在跑 测试交叉验证 2')
y_train_pred = cross_val_predict(rnd_clf, X_train, y_train_5, cv=10)   #每个数据预测值
y_score_train = cross_val_predict(rnd_clf, X_train, y_train_5, cv=10,method='predict_proba')   #每个数据预测得分
y_score_test = cross_val_predict(rnd_clf, X_test, y_test_5, cv=10,method='predict_proba')   #每个数据预测得分


print('预测类别:',y_train_pred)
print('预测得分:',y_train_pred)

print('交叉验证准确率:',cross_val_score(rnd_clf,X_train, y_train_5,cv=10,scoring='accuracy'))
print('交叉验证精度:',cross_val_score(rnd_clf,X_train, y_train_5,cv=10,scoring='precision'))
print('交叉验证召回率:',cross_val_score(rnd_clf,X_train, y_train_5,cv=10,scoring='recall'))
print('混淆矩阵:',confusion_matrix(y_train_5, y_train_pred))
print('auc:',roc_auc_score(y_test_5, y_score_test[:,1]))

