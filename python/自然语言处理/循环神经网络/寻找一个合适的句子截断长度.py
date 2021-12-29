# coding: utf-8


import glob
import os
import numpy as np
from random import shuffle
from nltk.tokenize import TreebankWordTokenizer
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def test_len(data, maxlen):
    total_len = truncated = exact = padded = 0
    for sample in data:
        total_len += len(sample)
        if len(sample) > maxlen:
            truncated += 1
        elif len(sample) < maxlen:
            padded += 1
        else:
            exact += 1
    print('填充的: {}'.format(padded))
    print('整好的: {}'.format(exact))
    print('截断的: {}'.format(truncated))
    print('平均长度: {}'.format(total_len / len(data)))

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
        with open(filename, 'r', encoding="utf-8") as f:
            dataset.append((pos_label, f.read()))

    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r', encoding="utf-8") as f:
            dataset.append((neg_label, f.read()))

    shuffle(dataset)

    return dataset


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

# 将标签单独拿出来
def collect_expected(dataset):
    """ Peel of the target values from the dataset """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


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


dataset = pre_process_data('D:/书籍资料整理/IMDB数据集/aclImdb/train')
word_vectors = KeyedVectors.load_word2vec_format(r'D:\数据集\谷歌新闻word2dev\GoogleNews-vectors-negative300.bin.gz',
                                                 binary=True, limit=200000)
vectorized_data = tokenize_and_vectorize(dataset)
#这个函数测得了数据集中被截断的,填充的,正好的,平均的句子词长度
# 我们使用平均句子词长度来建模(202)看下结果
test_len(vectorized_data, 400)

expected = collect_expected(dataset)
split_point = int(len(vectorized_data) * .8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]


# 神经网络参数
maxlen = 200
batch_size = 4  #只能减小批量换得内存的节约
embedding_dims = 300  # Length of the token vectors we will create for passing into the Convnet
# filters = 250           # Number of filters we will train
# kernel_size = 3         # The width of the filters, actual filters will each be a matrix of weights of size: embedding_dims x kernel_size or 50 x 3 in our case
# hidden_dims = 250       # Number of neurons in the plain feed forward net at the end of the chain
epochs = 2  # Number of times we will pass the entire training dataset through the network



# 这里还是用到了截断和填充.
# 因为貌似RNN以后的模型需要整齐的输入
# 因为前馈层需要整齐的输入


x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)



# 指的是50个隐藏层神经元
# 选择50个的原因是为了时间考虑
num_neurons = 50

print('Build model...')
model = Sequential()

# return_sequences=True相当于网络每个时刻都要返回网络输出,否则将返回最后一个时刻的50维向量
model.add(LSTM(num_neurons,
        return_sequences=True,
        input_shape=(maxlen, embedding_dims)))
# 指的是20%随机停止
# Dropout对于降低过拟合非常有用
# 但是需要限制在20%~50%
model.add(Dropout(.2))

# Flatten将rnn输出的400*50的张量转换为20000个原色的向量,便于sigmoid得出分类的结果
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# 一样无法在ipynotebook中训练,batch_size应该有个默认值可能是32
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))




model_structure = model.to_json()
with open("D:/中间结果/lstm_model2.json", "w") as json_file:
    json_file.write(model_structure)

model.save_weights("D:/中间结果/lstm_weights2.h5")
print('Model saved.')

#其实这里与元模型比降了2%的准确率,但是速度确实快很多.