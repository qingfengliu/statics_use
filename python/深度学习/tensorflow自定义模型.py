#1.张量操作
import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.constant([[1., 2., 3.], [4., 5., 6.]]) # 创建张量
print(tf.constant(42)) # 标量？
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t.shape)
print(t.dtype)
#操作方式很想numpy,举一下张量的运算操作
t + 10
tf.square(t)
#矩阵乘法(未知是点乘还是×乘)
t @ tf.transpose(t)

#向量运算还可以使用keras.backend keras可移植程序
K = keras.backend
K.square(K.transpose(t)) + 10
#与numpy相互转换
a = np.array([2., 4., 5.])
tf.constant(a)

t.numpy()
np.array(t)

#类型转换,类型转换很影响性能,所以tensorflow默认不做任何类型转换.
#如果不同类型的张量执行操作,会引发异常.如不能把浮点张量和整数张量相加,甚至不能把float32与float64相加
#如果要使用类型转换可以使用tf.cast()
t2 = tf.constant(40., dtype=tf.float64)
tf.constant(2.0) + tf.cast(t2, tf.float32)

#变量,tensor的值是不能变的,无法调整.很不方便,
#神经网络需要事实调整张量的值,需要使用
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
#可以使用assign()方法改变值,这里就不计了涉及定位问题.

#定制模型和训练算法
#(1)自定义损失函数
#以huber损失为例(适用于噪声较大的其实keras中已经实现)
#取加州房价数据
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

input_shape = X_train.shape[1:]

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])
#损失函数在compile函数中使用loss
model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))

#(2)保存和加载包含自定义组件的模型
#直接说结果,huber_fn里边有一个重要的参数，阈值,如果你保存模型这个值很难被保留下来,
#所以使用keras.losses.Loss类的子类主要其中get_config为保存阈值
#而call为损失函数计算函数

class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])
#注意阈值这里被定义为2
model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])

model.save("my_model_with_a_custom_loss_class.h5")
model = keras.models.load_model("my_model_with_a_custom_loss_class.h5",
                                custom_objects={"HuberLoss": HuberLoss})
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
print(model.loss.threshold) #这里打印出来损失函数还是2.0

#(3)自定义激活函数、初始化、正则化和约束
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)  #这段代码的功能未知
#自定义的各种

def my_softplus(z): # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)

layer = keras.layers.Dense(1, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)
#如果函数具有参数需要一并保存也需要像损失函数那样定义类
#对于损失函数、层、和模型实现call方法,或者正则化、初始化和约束实现__call__()方法
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1, activation=my_softplus,
                       kernel_regularizer=MyL1Regularizer(0.01),
                       kernel_constraint=my_positive_weights,
                       kernel_initializer=my_glorot_initializer),
])
model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
model.save("my_model_with_many_custom_parts.h5")
model = keras.models.load_model(
    "my_model_with_many_custom_parts.h5",
    custom_objects={
       "MyL1Regularizer": MyL1Regularizer,
       "my_positive_weights": my_positive_weights,
       "my_glorot_initializer": my_glorot_initializer,
       "my_softplus": my_softplus,
    })

#对于剩下的内容在我这个层次来说太难这里先不研究,如果有机会更深一步的学习可来学习
#(4)自定义指标
#(5)自定义层
#(6)自定义模型
#(7)基于模型内部的损失和指标
#(8)使用自动微分计算梯度
#(9)自定义训练循环

#Tensorflow的图,说白了就是使用tf将python的语句翻译一遍,有点解释器的意思
#如python函数
def cube(x):
    return x ** 3
#转换成tf函数
tf_cube = tf.function(cube)
#如果你不传入向量其实还是隐式的转换为向量
tf_cube(2)

#当你使用自定义函数,其实内部还是转换成了图
#但是如果将不同形状的张量,或者直接将不同的python数值
#传入则会使得tf生成并保存,不同的图,占用不必要的内存
tf_cube.python_function(2) #可以直接调用python函数而不是tf图

#另外可以使用修饰器来声明图转换如
@tf.function
def add_10(x):
    condition = lambda i, x: tf.less(i, 10)
    body = lambda i, x: (tf.add(i, 1), tf.add(x, 1))
    final_i, final_x = tf.while_loop(condition, body, [tf.constant(0), x])
    return final_x
#打印图结构
print(tf.autograph.to_code(add_10.python_function))

#最好不用外部库,如使用tf.reduce_sum()代替np.sum()并且如range函数也需要tf.range


