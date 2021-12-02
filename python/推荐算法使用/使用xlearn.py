import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xlearn as xl

iris_data = load_iris()

X = iris_data['data']
y = iris_data['target'] == 2

X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
linear_model = xl.LRModel(task='binary',init=0.1,epoch=10,lr=0.1,reg_lambda=1.0,opt='sgd')

linear_model.fit(X_train,y_train,eval_set=[X_test, y_test],is_lock_free=False)

y_pred = linear_model.predict(X_test)
