from nltk.corpus import gutenberg
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop

print(gutenberg.fileids())

text = ''
#将文章内容合并
for txt in gutenberg.fileids():
    if 'shakespeare' in txt:
        text += gutenberg.raw(txt).lower()

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
#字符串字典
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []

#了数据,每3个字符采样40个字符(在第6个字符再采样60个,依次类推)
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
#sentences是训练集,next_chars为标签,也就是说,监督学习模型,并且只预测下一个字母
#独热编码
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1



# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print(model.summary())

epochs = 6
batch_size = 128

model_structure = model.to_json()
with open("D:/中间结果/shakes_lstm_model.json", "w") as json_file:
    json_file.write(model_structure)

#每个6个周期保存中间结果？
for i in range(5):
    model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs)

    model.save_weights("D:/中间结果/shakes_lstm_weights_{}.h5".format(i + 1))
    print('Model saved.')


