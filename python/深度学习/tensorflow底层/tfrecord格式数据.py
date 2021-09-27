#tfrecord格式是一种二进制的格式,用于图像音频等比较合适
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.datasets import load_sample_images
#写文件
with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

#读数据
filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)

#可以一次读取多个文件，并且如果设置了num_parallel_reads可以并行读取多个文件的交织记录
filepaths = ["my_test_{}.tfrecord".format(i) for i in range(5)]
for i, filepath in enumerate(filepaths):
    with tf.io.TFRecordWriter(filepath) as f:
        for j in range(3):
            f.write("File {} record {}".format(i, j).encode("utf-8"))

dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=3)
for item in dataset:
    print(item)
    print('-----------------------------')

#建立压缩文件 这里就我觉得没必要例子展示了.在创建文件的时候与读取文件的时候都要加上
#compression_type="GZIP"
#协议缓冲区protobufs,google开发的一种协议.书中说需要装编译器,
#这是一个与平台和需要无关的东西,可以通过protobufs自带的工具解析成对应语言的数据结构
#感觉还挺方便的.



