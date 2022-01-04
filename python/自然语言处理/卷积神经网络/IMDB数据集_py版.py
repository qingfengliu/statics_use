#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.preprocessing import sequence    # 处理输入数据的辅助模块
from keras.models import Sequential         # 基础神经网络
from keras.layers import Dense, Dropout, Activation   # 常用层
from keras.layers import Conv1D, GlobalMaxPooling1D   #卷积层和池化层
import tensorflow as tf

# In[2]:


import glob
import os

from random import shuffle


os.environ["CUDA_VISIBLE_DEVICES"]="1"

#IMDB电影评论情感问题,是个二分类问题,0,1分布代表积极和消极
#文件组成,两个文件夹分别代表0和1
#每一个评论被做成了一个文件.
def pre_process_data(filepath):
    """
    This is dependent on your training data source but we will try to generalize it as best as possible.
    """
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'neg')

    pos_label = 1
    neg_label = 0

    dataset = []

    for filename in glob.glob(os.path.join(positive_path, '*.txt')):
        with open(filename, 'r',encoding="utf-8") as f:
            dataset.append((pos_label, f.read()))

    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r',encoding="utf-8") as f:
            dataset.append((neg_label, f.read()))

    shuffle(dataset)

    return dataset


dataset = pre_process_data(r'D:\书籍资料整理\IMDB数据集\aclImdb\train')
print(dataset[0])


# In[ ]:


from nltk.tokenize import TreebankWordTokenizer
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format(r'D:\数据集\谷歌新闻word2dev\GoogleNews-vectors-negative300.bin.gz',
                                                 binary=True, limit=100000)
#这里使用,谷歌新闻的word2dev,如果不在那里的词会被放弃.
#首先被限制了20W.并且还有一些停用词.导致处理后与原数据比起来有一些丢失

#此函数就是将所有评论分词并且将词向量解出来,这里标签被舍弃掉
def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    expected = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass  # No matching token in the Google w2v vocab

        vectorized_data.append(sample_vecs)

    return vectorized_data

#这里解出标签
def collect_expected(dataset):
    """ Peel of the target values from the dataset """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


# In[ ]:


vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)


# In[ ]:


#切分训练集,验证集
split_point = int(len(vectorized_data) * .8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]


# In[ ]:


#神经网络参数,可以看到最大长度为400，那么需要超过400的阶段,低于400的填充.

maxlen = 400        #
batch_size = 32         # 小批量,批量数
embedding_dims = 300    # 词向量的长度,影响神经网络结构形状
filters = 250           # 卷积核数量
kernel_size = 3         # 卷积核大小
hidden_dims = 250       # Number of neurons in the plain feed forward net at the end of the chain
epochs = 2              # Number of times we will pass the entire training dataset through the network


# In[ ]:


# 填充/和截断每个句子

def pad_trunc(data, maxlen):
    """ For a given dataset pad with zero vectors or truncate to maxlen """
    new_data = []

    # Create a vector of 0's the length of our word vectors
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:

        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data


# In[ ]:


x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)


# In[ ]:


x_train = np.reshape(np.array(x_train,dtype='float32'), (len(x_train), maxlen, embedding_dims))
print(x_train.shape)
y_train = np.array(y_train,dtype='float32')
x_test = np.reshape(np.array(x_test,dtype='float32'), (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test,dtype='float32')


# In[ ]:



print('Build model...')
model = Sequential()
#keras卷积层,一维卷积
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 input_shape=(maxlen, embedding_dims)))


# In[ ]:


#使用了全局最大池化层很大程度上丢失数据,但是书中说这个例子没问题
#注意这里是在卷积核之后的一层,最大池化的是卷积的输出
model.add(GlobalMaxPooling1D())
#keras全连接层
#添加全连接神经网络,学习卷积层过来的特征
#并将特征映射到隐向量空间
#默认250个神经元 通过hiden_dims调节
model.add(Dense(hidden_dims))

# keras Dropout用于防止过拟合,梯度消失梯度爆炸的技术,
# 通过在训练时 在某轮停止某些神经元来做到
model.add(Dropout(0.2))
model.add(Activation('relu'))

#输出层,本例为二分类问题,
model.add(Dense(1))
model.add(Activation('sigmoid'))

#编译CNN,损失函数binary_crossentropy 因为是二分类
# categorical_crossentropy 用于多分类
#adam优化器
#
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:



model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


# In[ ]:

model_structure = model.to_json()
with open("D:/中间结果/cnn_model.json", "w") as json_file:
    json_file.write(model_structure)

model.save_weights("D:/中间结果/cnn_weights.h5")
print('Model saved.')

