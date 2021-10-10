import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.random.set_seed(42)
#使用imdb数据集(影评,分析是正向情感还是负向的)
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
print(X_train.shape)
#数据集已经被处理,全部小写 删除标点符号 空格分隔
#按频率索引,0 1 2是特殊的分别代表填充令牌 序列开始 未知单词
#对评率进行还原

word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ + 3: word for word, id_ in word_index.items()}

for id_, token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token
" ".join([id_to_word[id_] for id_ in X_train[0][:10]])

#有时候我们需要加载原始数据集,然后自己预处理
import tensorflow_datasets as tfds
datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)

# datasets.keys()
# train_size = info.splits["train"].num_examples
# test_size = info.splits["test"].num_examples

#打印内容
# for X_batch, y_batch in datasets["train"].batch(2).take(1):
#     for review, label in zip(X_batch.numpy(), y_batch.numpy()):
#         print("Review:", review.decode("utf-8")[:200], "...")
#         print("Label:", label, "= Positive" if label else "= Negative")
#         print()

#处理函数1:截断取前300个单词,并且将某些字符替换成空格,最后切割字符并且向量化(不同的长度会填充<pad>)
def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch


#使用Counter结构统计词频
from collections import Counter
vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))

print(vocabulary.most_common()[:3]) #打印词频最高的三个词

#取最频繁的10000个单词
vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]

# 加上1000 out-of-vocabulary

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

print(table.lookup(tf.constant([b"This movie was faaaaaantastic".split()])))

def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

# 准备最终数据 首先preprocess转换短序列
# 然后encode_words进行编码 该函数使用刚才构建的词汇表
train_set = datasets["train"].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

for X_batch, y_batch in train_set.take(1):
    print(X_batch)
    print(y_batch)

#创建模型训练
#第一层是嵌入层,将单词ID转换为嵌入也就是说
embed_size = 128
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                           mask_zero=True, # not shown in the book
                           input_shape=[None]),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, epochs=5)

#掩码屏蔽mask，接收时间步长的层都会被保留mask掩码,所有能接收到掩码的层都必须能处理掩码否则将报错
#下边第二个GRU层,将不再返回掩码
K = keras.backend
embed_size = 128
inputs = keras.layers.Input(shape=[None])
mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
z = keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(inputs)
z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
z = keras.layers.GRU(128)(z, mask=mask)
outputs = keras.layers.Dense(1, activation="sigmoid")(z)
model = keras.models.Model(inputs=[inputs], outputs=[outputs])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_set, epochs=5)

#使用预训练过的模型

#可能被墙了
import tensorflow_hub as hub

model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                   dtype=tf.string, input_shape=[], output_shape=[50]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

import tensorflow_datasets as tfds

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples
batch_size = 32
train_set = datasets["train"].batch(batch_size).prefetch(1)
history = model.fit(train_set, epochs=5)

