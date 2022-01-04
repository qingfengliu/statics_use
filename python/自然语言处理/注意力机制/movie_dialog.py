import os
from nlpia.loaders import get_data
import numpy as np  # <1> # noqa
from keras.models import Model  # noqa
from keras.layers import Input, LSTM, Dense  # noqa

os.environ["CUDA_VISIBLE_DEVICES"]="1"
df = get_data('moviedialog')

df.columns = 'statement reply'.split()
df = df.dropna()
input_texts, target_texts = [], []  # <1>
start_token, stop_token = '\t\n'  # <3>
df['target'] = start_token + df.reply + stop_token

input_vocab = set()  # <2>
output_vocab = set(start_token + stop_token)
n_samples = min(100000, len(df))  # <4>

#建立 字典
[input_vocab.update(set(statement)) for statement in df.statement]
[output_vocab.update(set(reply)) for reply in df.reply]
input_vocab = tuple(sorted(input_vocab)) #<6>
output_vocab = tuple(sorted(output_vocab))


#最长的字符串长度
max_encoder_seq_len = df.statement.str.len().max()
# max_encoder_seq_len
# 100
max_decoder_seq_len = df.target.str.len().max()

#独热编码部分
encoder_input_onehot = np.zeros(
    (len(df), max_encoder_seq_len, len(input_vocab)),
    dtype='float32')  # <2>

#解码器输入向量是比解码器输出向量后移一个时序
decoder_input_onehot = np.zeros(
    (len(df), max_decoder_seq_len, len(output_vocab)),
    dtype='float32')
decoder_target_onehot = np.zeros(
    (len(df), max_decoder_seq_len, len(output_vocab)),
    dtype='float32')

for i, (input_text, target_text) in enumerate(
        zip(df.statement, df.target)):  # <3>
    for t, c in enumerate(input_text):  # <4>
        k = input_vocab.index(c)
        encoder_input_onehot[i, t, k] = 1.  # <5>
    k = np.array([output_vocab.index(c) for c in target_text])
    decoder_input_onehot[i, np.arange(len(target_text)), k] = 1.
    decoder_target_onehot[i, np.arange(len(target_text) - 1), k[1:]] = 1.


#
batch_size = 64    # <1>
epochs = 100       # <2>
num_neurons = 256  # <3>

#构建神经网络部分使用的是函数式API
#另一种形式是Sequential() 只要将模型ADD进去就行了相当于模板但是无法适用于所有情况
#
#编码器 构建过程
encoder_inputs = Input(shape=(None, len(input_vocab)))
encoder = LSTM(num_neurons, return_state=True)
#state_h是这个层是最后一个时刻的具体输出 state_c是记忆单元状态
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#encoder_states是思想向量 用于解码器的记忆向量
encoder_states = [state_h, state_c]
# 解码器构建
decoder_inputs = Input(shape=(None, len(output_vocab)))
decoder_lstm = LSTM(num_neurons, return_sequences=True,
                    return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
# 解码器的lstm后跟一个全连接层
decoder_dense = Dense(len(output_vocab), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 看到这里model的建立,第一个参数为 输入的数据序列,最后一个参数为 解码器接的全连接层
# decoder_outputs 虽然不知道内部原理但是一定是通过 每个结构的输出参数将
# 整个结构联系在一起,函数式API是很高级的实现方式了
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 多分类损失函数

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['acc'])

model.fit([encoder_input_onehot, decoder_input_onehot],
          decoder_target_onehot, batch_size=batch_size, epochs=epochs,
          validation_split=0.1)  # <4>

#专用于输出的结构
#这里与上边定义的形式略有不同 将编码器解码器分开定义model
#编码器的任务是输入文本产生思想向量
#解码器根据思想向量来产生输出
# 编码器结构 仅是输入 encoder_inputs 输出为encoder_states
# 这里看着显的突兀, 实际上与训练结构相比基本没变
# 上边已经定义好 这里比较神奇的魔法是 直接使用输入和输出就直接将
# 前边定义好的结构联系在了一起十分神奇

encoder_model = Model(encoder_inputs, encoder_states)

# 解码器结构
# thought_input思想向量单独声明了一下结构 与上边的训练模型不同
# 因为编码器和解码器被分开声明的缘故
thought_input = [
    Input(shape=(num_neurons,)), Input(shape=(num_neurons,))]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=thought_input)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    inputs=[decoder_inputs] + thought_input,
    output=[decoder_outputs] + decoder_states)

#输入
def decode_sequence(input_seq):
    #生成思想向量作为解码器的输入
    thought = encoder_model.predict(input_seq)  # <1>

    target_seq = np.zeros((1, 1, len(output_vocab)))  # <2>
    target_seq[0, 0, output_vocab.index(stop_token)
        ] = 1.  # <3>
    stop_condition = False
    generated_sequence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + thought) # <4>

        generated_token_idx = np.argmax(output_tokens[0, -1, :])
        generated_char = output_vocab[generated_token_idx]
        generated_sequence += generated_char
        if (generated_char == stop_token or
                len(generated_sequence) > max_decoder_seq_len
                ):  # <5>
            stop_condition = True

        target_seq = np.zeros((1, 1, len(output_vocab)))  # <6>
        target_seq[0, 0, generated_token_idx] = 1.
        thought = [h, c]  # <7>

    return generated_sequence


def respond(input_text):
    input_text = input_text.lower()
    input_text = ''.join(c if c in input_vocab else ' ' for c in input_text)
    input_seq = np.zeros((1, max_encoder_seq_len, len(input_vocab)), dtype='float32')
    for t, c in enumerate(input_text):
        input_seq[0, t, input_vocab.index(c)] = 1.
    decoded_sentence = decode_sequence(input_seq)
    print('Human: {}'.format(input_text))
    print('Bot:', decoded_sentence)
    return decoded_sentence

respond('Hi Rosa, how are you?')
respond('Hi Jim, how are you?')