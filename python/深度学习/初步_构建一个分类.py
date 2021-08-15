import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pydoc
import graphviz
import numpy as np
print(tf.__version__)
print(keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#在使用梯度下降计算神经网络时,需要进行特征缩放.为了简单直接/255,将像素降到0~1的浮点数
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

#展示了一张图
# plt.imshow(X_train[0], cmap="binary")
# plt.axis('off')
# plt.show()

#这数组对应了,数据集中的书中类标签class_names[y_train[0]] 就可以差标签对应的类名
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])  #
# model.summary()  #查看各层信息

keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
#创建模型后,必须调用complie()方法指定损失函数和要使用的优化器
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
#使用sparse_categorical_crossentropy因为我们有稀疏数据,我猜意思为(标签0-9)
#如果是独热向量(onehotcode)则使用categorical_crossentropy
#如果是二分类,则使用binary_crossentropy并且激活函数为sigmoid
#如果将稀疏标签(即类索引)转换为独热向量则使用keras.utils.to_categorical()


history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
#传递一个验证集可以帮助我们判断是否过拟合很有帮助epochs训练论次

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

plt.show()

#让我们跳过书中繁琐的过程直接来看预测步骤
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))  #预测概率矩阵
y_pred = np.argmax(model.predict(X_new), axis=-1)
print(y_pred)  #预测结果

