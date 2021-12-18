import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# 基于Keras的API的模型定义需要继承Model类，重写方法call（前向传播过程），
# 如果需要加入dropout的模型需要将训练和预测分开可以使用参数training=None的方式来指定，
# 这里不需要这个参数，因此省略。
class AutoRec(keras.Model):
    def __init__(self, feature_nums, hidden_units, **kwargs):
        super(AutoRec, self).__init__()
        self.feature_nums = feature_nums # 基于物品则为物品特征数-即用户数，基于用户则为物品数量
        self.hidden_units = hidden_units # 论文中的k参数
        self.encoder = keras.layers.Dense(self.hidden_units, input_shape=[self.feature_nums], activation='sigmoid') # g映射
        self.decoder = keras.layers.Dense(self.feature_nums, input_shape=[self.hidden_units]) # f映射

    def call(self, X):
        # 前向传播
        h = self.encoder(X)
        y_hat = self.decoder(h)
        return y_hat

# 此损失函数虽然为MSE形式，但是在计算的过程中发现，仅仅计算有评分的部分，无评分部分不进入损失，同时还有正则化，这里一起写出来。
# 基于Keras API的方式，需要继承Loss类，和方法call初始化传入model参数为了取出W和V参数矩阵。
# mask_zero表示没有评分的部分不进入损失函数，同时要保证数据类型统一tf.int32,tf.float32否则会报错。
class Mse_Reg(keras.losses.Loss):
    def __init__(self, model, reg_factor=None):
        super(Mse_Reg, self).__init__()
        self.model = model
        self.reg_factor = reg_factor

    def call(self, y_true, y_pred):
        y_sub = y_true - y_pred
        mask_zero = y_true != 0
        mask_zero = tf.cast(mask_zero, dtype=y_sub.dtype)
        y_sub *= mask_zero
        mse = tf.math.reduce_sum(tf.math.square(y_sub))  # mse损失部分
        reg = 0.0
        if self.reg_factor is not None:
            weight = self.model.weights
            for w in weight:
                if 'bias' not in w.name:
                    reg += tf.reduce_sum(tf.square(w))  # 求矩阵的Frobenius范数的平方
            return mse + self.reg_factor * 0.5 * reg
        return mse

# 定义评价指标需要继承类Metric，方法update_state和result以及reset,reset方法感觉使用较少，主要是更新状态和得到结果。
class RMSE(keras.metrics.Metric):
    def __init__(self):
        super(RMSE, self).__init__()
        self.res = self.add_weight(name='res', dtype=tf.float32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_sub = y_true - y_pred
        mask_zero = y_true != 0
        mask_zero = tf.cast(mask_zero, dtype=y_sub.dtype)
        y_sub *= mask_zero
        values = tf.math.sqrt(tf.reduce_mean(tf.square(y_sub)))
        self.res.assign_add(values)

    def result(self):
        return self.res

# get_data表示从path中加载数据，然后加数据通过pandas的透视表功能构造一个行为物品，列为用户的矩阵；
# data_iter表示通过tf.data构造数据集。
# 定义数据
def get_data(path, base_items=True):
    data = pd.read_csv(path)
    rate_matrix = pd.pivot_table(data, values='rating', index='movieId', columns='userId', fill_value=0.0)
    if base_items:
        return rate_matrix
    else:
        return rate_matrix.T


def data_iter(df, shuffle=True, batch_szie=32, training=False):
    df = df.copy()
    X = df.values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((X, X)).batch(batch_szie)
    if training:
        ds = ds.repeat()
    return ds

path = 'D:/model/ml-latest-small/ratings.csv' # 我这里用的是10w数据，不是原始的movielens-1m
# I-AutoRec，num_users为特征维度
rate_matrix = get_data(path)
num_items, num_users = rate_matrix.shape

# 划分训练测试集
BARCH = 128
train, test = train_test_split(rate_matrix, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)
train_ds = data_iter(train, batch_szie=BARCH, training=True)
val_ds = data_iter(val, shuffle=False)
test_ds = data_iter(test, shuffle=False)

# 定义模型
net = AutoRec(feature_nums=num_users, hidden_units=500) # I-AutoRec, k=500
net.compile(loss=Mse_Reg(net), #keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(),
            metrics=[RMSE()])
net.fit(train_ds, validation_data=val_ds, epochs=40, validation_steps=2, steps_per_epoch=train.shape[0]//BARCH)

loss, rmse = net.evaluate(test_ds)
print('loss: ', loss, ' rmse: ', rmse)

#预测
df = test.copy()
X = df.values.astype(np.float32)
ds = tf.data.Dataset.from_tensor_slices(X) # 这里没有第二个X了
ds = ds.batch(32)
pred = net.predict(ds)
# 随便提出来一个测试集中有的评分看看预测的分数是否正常,pred包含原始为0.0的分数现在已经预测出来分数的。
print('valid: pred user1 for item1: ', pred[1][X[1].argmax()], 'real: ', X[1][X[1].argmax()])