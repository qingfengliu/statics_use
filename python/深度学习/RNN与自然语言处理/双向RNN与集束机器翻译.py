import tensorflow as tf

#双向RNN同一输入上循环运行两个循环层,一层从左至右读取单词,另一个层从右至左读取单词.然后,只需在每个时间步长上简单地组合它们的输出,通常将他们合并即可.

from tensorflow import keras
import tensorflow_addons as tfa

model = keras.models.Sequential([
    keras.layers.GRU(10, return_sequences=True, input_shape=[None, 10]),
    keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences=True))
])

model.summary()

#集束模型,我认为这篇翻译有问题集束的意思是.首先对这个单词,我们会返回他的是某个翻译的单词的概率,然后在看下一个单词求组合概率,以此类推
#但是这个方法对长句子还是无法取得良好效果.这里貌似代码不全待补充

beam_width=10
decoder=tfa.seq2seq.beam_search_decoder.BeamSearchDecoder(cell=decoder_cell,beam_width=beam_width,output_layer=output_layer)
deecoder_initial_state=tfa.seq2seq.beam_search_decoder.tile_batch(encoder_state,multiplier=beam_width)
outputs,_,_ = decoder(embedding_decoder,start_tokens=start_tokens,end_token=end_token,initial_state=deecoder_initial_state)


