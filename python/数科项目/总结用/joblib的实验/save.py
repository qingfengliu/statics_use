import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import joblib


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
joblib.dump(rnd_clf, "my_model.pkl")
joblib.dump(X_test, "x_test.pkl")
joblib.dump(y_test_5, "y_test.pkl")