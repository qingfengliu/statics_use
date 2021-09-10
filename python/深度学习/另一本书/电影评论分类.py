from  keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

#保留训练数据中前10000个最常见
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)



x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#输入数据是标量0,1有一类网络在这种问题上表现良好.
#就是带有relu激活的全连接层
#每个rule激活的全连接层都实现了下列张量运算
#output=relu(dot(W,input)+b)
#16个隐藏单元可以理解为网络学习内部表示时所拥有的自由度.
#隐藏单元越多能学到的更加复杂,但是可能会导致过拟合
#对于Dense层堆叠,需要回答两个关键问题确定架构
#网络有多少层,每层有多少个隐藏单元.
#这里那么使用现成的,两个中间层有16个隐藏单元.第三层输出一个标量,预测当前评论的感情.
#中间层使用relu做激活函数,最后一层使用sigmoid激活

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#val指的是验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#使用512个样本组成的小批量,将模型训练20次.每轮结束时会有短暂停顿.
#因为模型要计算在验证集的10000个样本上的损失和精度
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
#history 包含训练过程中的所有数据
history_dict = history.history
print(history_dict.keys())



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

#训练损失和验证损失

epochs = range(1, len(acc) + 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')  # bo表示蓝色圆点
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')  # b表示蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()  #如果不加这一句就不会显示图例
plt.show()

#绘制训练训练精度和验证精度

plt.clf()   # clear figure
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#验证集的精度差不多,损失不断升高,这种情况来说就是在训练集损失降低,验证集损失不断升高,模型存在过拟合.

#1.带有relu激活的Dense层堆叠,可以解决很多问题,(包括情感分析).
#2.对于二分类问题,网络的最后一层应该是只有一个单元并使用sigmoid激活的dense层,网络输出应该是0~1范围内的标量,表示概率值
#3.无论什么问题,rmsprop优化器通常都是足够好的选择.这点无需担心.
#4.神经网络在训练数据上的表现越来越好,模型最终会过拟合,并在前所未见的数据上得到越来越差的结果.一定监控模型之外的数据的性能.（也就是说需要验证集？）
#5.对于二分类问题的sigmoid标量输出应该使用binary_crossentroy损失函数.


