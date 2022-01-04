import glob
import os

from random import shuffle
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from gensim.models import KeyedVectors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

def collect_expected(dataset):
    """ Peel of the target values from the dataset """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected

#计算平均字符长(这里指每段字符数的总数的均值)
def avg_len(data):
    total_len = 0
    for sample in data:
        total_len += len(sample[1])
    print(total_len / len(data))

#清洗数据,如果不在VALID中,则替换为UNK表示为未知字符(将字符串都变成了小写了)
def clean_data(data):
    """ Shift to lower case, replace unknowns with UNK, and listify """
    new_data = []
    VALID = 'abcdefghijklmnopqrstuvwxyz123456789"\'?!.,:; '
    for sample in data:
        new_sample = []
        for char in sample[1].lower():  # Just grab the string, not the label
            if char in VALID:
                new_sample.append(char)
            else:
                new_sample.append('UNK')

        new_data.append(new_sample)
    return new_data

#这里字符串填充和截断,不够的maxlen的将被填充为PAD
def char_pad_trunc(data, maxlen):
    """ We truncate to maxlen or add in PAD tokens """
    new_dataset = []
    for sample in data:
        if len(sample) > maxlen:
            new_data = sample[:maxlen]
        elif len(sample) < maxlen:
            pads = maxlen - len(sample)
            new_data = sample + ['PAD'] * pads
        else:
            new_data = sample
        new_dataset.append(new_data)
    return new_dataset

#字符模型的字典
def create_dicts(data):
    """ Modified from Keras LSTM example"""
    chars = set()
    for sample in data:
        chars.update(set(sample))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return char_indices, indices_char

#建立字符串的独热编码
def onehot_encode(dataset, char_indices, maxlen):
    """
    One hot encode the tokens

    Args:
        dataset  list of lists of tokens
        char_indices  dictionary of {key=character, value=index to use encoding vector}
        maxlen  int  Length of each sample
    Return:
        np array of shape (samples, tokens, encoding length)
    """
    X = np.zeros((len(dataset), maxlen, len(char_indices.keys())))
    for i, sentence in enumerate(dataset):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
    return X

dataset = pre_process_data('D:/书籍资料整理/IMDB数据集/aclImdb/train')
expected = collect_expected(dataset)

expected=np.array(expected)
print(avg_len(dataset))
listified_data = clean_data(dataset)
maxlen = 1500
common_length_data = char_pad_trunc(listified_data, maxlen)

#生成索引字典
char_indices, indices_char = create_dicts(common_length_data)

#onehot_encode onehot必须按照索引字典创建,否则索引字典无意义,我们
#onehot之后的代码无法还原回词也是无意义
encoded_data = onehot_encode(common_length_data, char_indices, maxlen)
split_point = int(len(encoded_data) * .8)
x_train = encoded_data[:split_point]
y_train = expected[:split_point]
x_test = encoded_data[split_point:]
y_test = expected[split_point:]



from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, LSTM
num_neurons = 40

print('Build model...')
model = Sequential()

model.add(LSTM(num_neurons, return_sequences=True, input_shape=(maxlen, len(char_indices.keys()))))
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile('rmsprop', 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())

batch_size = 32
epochs = 10
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
model_structure = model.to_json()
with open("D:/中间结果/char_lstm_model3.json", "w") as json_file:
    json_file.write(model_structure)

model.save_weights("D:/中间结果/char_lstm_weights3.h5")
print('Model saved.')

#模型整体来说存在严重过拟合
#基于字符串的模型,使用的是onehot
#与词模型处理的时候有明显的两个差异
#1.create_dicts函数 生成字符串字典
#2.onehot_encode函数 onehot编码部分
