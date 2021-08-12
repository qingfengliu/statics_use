#MINIST,数据集为一组由美国高中生和人口调查局员工手写的70000个数字的图片。被广泛使用,初步涉猎机器学习可用这个数据集测试
#首先引入数据集
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
# print(mnist.keys())
X, y = mnist["data"], mnist["target"]
# print(type(X))
# print(X.shape)
# print(y.shape)
y = y.astype(np.uint8)

some_digit = X[0]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap=mpl.cm.binary)
# plt.axis("off")

# save_fig("some_digit_plot")
# plt.show()

#划分训练集与测试集,MNIST实际已经划分好了
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#尝试训练一个二分类器,比如5为1,其他的值为0.
#下边这种写法可以快速把标记转换为true和false的布尔变量

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#使用随机梯度下降法
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)
# print(sgd_clf.predict([some_digit]))

#性能验证
#(1)交叉验证测试准确率
#3折交叉验证
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

#3折交叉验证的准确率为百分之93%.但是这是因为只有10%的图片是5.所以准确率无法成为首要的指标
#尤其是判断有偏数据集时
#二分类问题混淆矩阵.但是求混淆矩阵前,需要一组预测，可以执行cross_val_predict获得

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print('y_train_pred:',y_train_pred)
#求混淆矩阵
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

#混淆矩阵格式
#   TN(真负例)  FP(假正例)
#   FN(假负例)  TP(真正例)

#求精度和召回率
from sklearn.metrics import precision_score, recall_score

print('precision score:',precision_score(y_train_5, y_train_pred))
print('recall score',recall_score(y_train_5, y_train_pred))

#很遗憾和书中结果不一致,精度为83%   召回率为65%. 表示 当预测一张图片是5的时候有83%是准确的
#并且仅有65%的5被检测出

#f1只有当召回率和精度都很高时,分类器才能得到较高的f1分数
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

#SDG可以通过调整阈值来在精度与准度之间权衡
#SDG给每个预测点一个评分,我们可以通过decision_function查看这个评分
y_scores = sgd_clf.decision_function([some_digit])
print('some_digit score:',y_scores)
#假设阈值为0,判断
threshold = 0
y_some_digit_pred = (y_scores > threshold)

#下边看如何选一个合适的阈值，首先我们使用cross_val_predict(),获得所有训练的实例分数.
#然后使用precision_recall_cuvrve()函数获得所有可能阈值的精度和召回率,然后作图看看情况
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()
#根据图精度会有抖动,原因可在书中P95找
#并且可以设定精度为90%而得到阈值
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] #精度90%
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]   #在精度为90%时召回率
# print(threshold_90_precision)
plt.figure(figsize=(8, 4))                                                                  # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")   #竖线              # Not shown
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")           #横线最上边的                     # Not shown
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")  #横线下边那条 # Not shown
plt.plot([threshold_90_precision], [0.9], "ro")                                             # Not shown
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown
# save_fig("precision_recall_vs_threshold_plot")                                              # Not shown
plt.show()

#也可以使用这种精度与召回率的联合图
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
plt.show()

#最后我们可以用如下函数来在一边看我们找的精度90%下的精度和召回率
y_train_pred_90 = (y_scores >= threshold_90_precision)

precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

#ROC曲线 首先需要使用roc_curve()函数计算多种阈值的TPR和FPR
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
#绘制ROC曲线函数
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)

plt.figure(figsize=(8, 6))                                    # Not shown
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           # Not shown
# print(recall_90_precision,fpr_90)
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   # Not shown
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  # Not shown
plt.plot([fpr_90], [recall_90_precision], "ro")               # Not shown
plt.show()

#计算AUC曲线的面积
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)

# 随机森林与其对比。
# 与SDG相比随机森林给出的不是评分而是数据类的概率.列表示每个分类的概率
# 行表示每个数据
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
plt.plot([fpr_90], [recall_90_precision], "ro")
plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:")
plt.plot([fpr_90], [recall_for_forest], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)
plt.show()


roc_auc_score(y_train_5, y_scores_forest)

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
precision_score(y_train_5, y_train_pred_forest)  #随机森林精度

recall_score(y_train_5, y_train_pred_forest)   #随机森林召回率