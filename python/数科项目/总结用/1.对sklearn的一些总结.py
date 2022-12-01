import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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


y_pred_train = rnd_clf.predict(X_train)  #在训练集上预测
y_pred_test = rnd_clf.predict(X_test)  #在测试集上预测
y_score_test =rnd_clf.predict_proba(X_test)   #测试集每个点的预测得分,可用于计算AUC

feat_labels=X_train.columns
importance=rnd_clf.feature_importances_   #随机森林能够求出特征重要性,然而这个图像例子里并不适用
imp_result=np.argsort(importance)[::-1]
for j,i in enumerate(imp_result):
    print("%2d. %-*s %f" %(j+1,30,feat_labels[i],importance[i]))


print('训练集准确率:',accuracy_score(y_train_5, y_pred_train))
print('训练集精度:',precision_score(y_train_5, y_pred_train))
print('训练集召回率:',recall_score(y_train_5, y_pred_train))

print('测试集准确率:',accuracy_score(y_test_5, y_pred_test))
print('测试集精度:',precision_score(y_test_5, y_pred_test))
print('测试集召回率:',recall_score(y_test_5, y_pred_test))

print('F1:',f1_score(y_test_5, y_pred_test))
print('auc:',roc_auc_score(y_test_5, y_score_test[:,1]))
#混淆矩阵格式
#   TN(真负例)  FP(假正例)
#   FN(假负例)  TP(真正例)

print('训练集混淆矩阵:',confusion_matrix(y_train_5, y_pred_train))
print('测试集混淆矩阵:',confusion_matrix(y_test_5, y_pred_test))

#混淆矩阵格式
#   TN(真负例)  FP(假正例)
#   FN(假负例)  TP(真正例)

precisions, recalls, thresholds = precision_recall_curve(y_test_5, y_score_test[:,1])


#召回率—精度阈值
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

fpr, tpr, thresholds = roc_curve(y_test_5, y_score_test[:,1])
plt.figure(figsize=(8, 6))                                    # Not shown
plot_roc_curve(fpr, tpr)
plt.show()

plot_learning_curves(rnd_clf, X, y)

