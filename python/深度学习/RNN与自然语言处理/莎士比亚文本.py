import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

#读取输入数据.属于char-rnn模型

filepath = r'D:\书籍资料整理\文本处理\莎士比亚文本.txt'
with open(filepath) as f:
    shakespeare_text = f.read()

#分词转换器,可以找到文本中的所有字符并将其转换为从1开始的数字.char_level=True将字符转换为小写
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)
print(tokenizer.texts_to_sequences(["First"]))
print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))
max_id = len(tokenizer.word_index) # 有多少不重复的字符
dataset_size = tokenizer.document_count # 一共有多少个字符
print(max_id,dataset_size)

#对全文编码
[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1

#手动的切分了90%作为训练集
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

print(encoded)
#使用windows方法将长字符序列转换为不重叠的窗口.?这里不重叠？从结果看貌似是重叠的
#间隔为1,窗口长度为101
n_steps = 100
window_length = n_steps + 1 # target = input shifted 1 character ahead
dataset = dataset.window(window_length, shift=1, drop_remainder=True)

#window方法返回的格式并不是张量需要调用flat_map变成张量
dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)

#将前100个字符与目标字符分开
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

#对数据结构中的每个结构进行热独编码
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

dataset = dataset.prefetch(1)  #预取

#建立char-Rnn模型
#包含两个GRU层,包含128个单元.dropout率为20%,输出层是一个时间分布的Dense层该层必须有39个单元
model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                     #dropout=0.2, recurrent_dropout=0.2),
                     dropout=0.2),
    keras.layers.GRU(128, return_sequences=True,
                     #dropout=0.2, recurrent_dropout=0.2),
                     dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                    activation="softmax"))
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
history = model.fit(dataset, epochs=10)

#预测部分,首先将要预测的数据处理一下
def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

X_new = preprocess(["How are yo"])
#Y_pred = model.predict_classes(X_new)
Y_pred = np.argmax(model(X_new), axis=-1)
print(tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]) # 1st sentence, last char

#生成莎士比亚文本.
#方法原理大致为提供给一些文本,模型预测,然后将预测值添加到文本末尾.再送入模型
#但是这种方式会使得输出在几个字母间重复.可以使用tf.random.category()引入随机性
#另外函数中的temperature是通用的,接近0的温度倾向于高概率的字符
def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


tf.random.set_seed(42)

print(complete_text("t", temperature=0.2))
print(complete_text("t", temperature=1))
print(complete_text("t", temperature=2))