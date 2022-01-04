from nltk.corpus import gutenberg
import numpy as np
from keras.models import model_from_json
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
text = ''
#将文章内容合并
for txt in gutenberg.fileids():
    if 'shakespeare' in txt:
        text += gutenberg.raw(txt).lower()
chars = sorted(list(set(text)))
print('total chars:', len(chars))
#字符串字典
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
with open("D:/中间结果/shakes_lstm_model.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)

model.load_weights('D:/中间结果/shakes_lstm_weights_5.h5')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # 应该是将 概率值 变换分布 以能够 控制 生成文本效果
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[ ]:


import sys
maxlen = 40
start_index = random.randint(0, len(text) - maxlen - 1)

for diversity in [0.2, 0.5, 1.0]:
    print()
    print('----- diversity:', diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()

#这个生成文本有些问题 生成内容不受控看下一章是怎么处理这个问题的。