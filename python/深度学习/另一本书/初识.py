from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])   #建立神经网络,Dense描述的是层
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#在开始之前需要对数据进行预处理,将其变成网络要求的形状,并且缩放到0,1区间.

#
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(predictions[0])
print(predictions[0][7])
print(test_labels[0])

#模型评估
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")

#关于张量
#向量(一维张量)
x = np.array([12, 3, 6, 14, 7])

#矩阵(二维张量)
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])

#三维张量
x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])

#x.ndim  #轴数
#train_images.shape  #形状
#train_images.dtype  #数据类型

# 切片操作
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
my_slice = train_images[10:100, :, :]  # 比较推荐这种切片方式,更直观
my_slice.shape

my_slice = train_images[:, 14:, 14:]  # 每个图片右下角截取14*14个像素

my_slice = train_images[:, 7:-7, 7:-7] # 图像中心截取14*14个像素

# 批量操作,通常来说所有数据张量的第一个轴,都是样本轴.在MNIST的例子中,样本就是数字图像
# 深度学习模型不会同时处理整个数据集,而是将数据拆解成小批量.

batch=train_images[:128]
batch=train_images[128:256]

n = 3
batch = train_images[128 * n:128 * (n + 1)]
x = np.array([[0., 1.],
             [2., 3.],
             [4., 5.]])
print(x.shape)
x = x.reshape((6, 1))
print(x)
x = np.zeros((300, 20))
#转置
x = np.transpose(x)
print(x.shape)

