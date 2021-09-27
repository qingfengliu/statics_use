import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
assert tf.__version__ >= "2.0"
from tensorflow import keras
import numpy as np
import os

#1.创建数据集
X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
# print(dataset)
dataset = tf.data.Dataset.range(10)
#可以说创建了一个一维张量
#遍历
for item in dataset:
    print(item)
print('----------------------------------------------')
dataset = dataset.map(lambda x: x * 2) #与pandas作用相似每个元素*2

#shuffle
tf.random.set_seed(42)

dataset = tf.data.Dataset.range(10).repeat(3)
#这里也是个一维张量共30个元素


#从dataset这个Buff中(共30个元素),不放回的抽取,变成了二维张量,第二个轴上为7个元素,
#知道抽完所以生成的二维张量一共30个元素
dataset = dataset.shuffle(buffer_size=3, seed=42).batch(7)
for item in dataset:
    print(item)

#加载加州住房数据将其分割为测试集和验证集
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_

def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = os.path.join("datasets", "housing")
    os.makedirs(housing_dir, exist_ok=True)
    path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths

#这里分割文件以便演示,大型数据集的文件级的乱序
train_data = np.c_[X_train, y_train]
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)
print('---------------------------------------------------------------')
#分割文件并生成文件路径
train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)
print(train_filepaths)

#创建文件读取管道
#返回一个乱序的文件路径,有时候你可能不需要这样那么可以设置shuffle=False
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)

#取出5个乱序的文件,并跳过文件头.interleave()默认并不并行读取,可以使用num_parallel_calls参数设置所需线程数
#甚至可以设置成tf.data.experimental.AUTOTUNE,使CPU动态选择合适的线程数
n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers)

for line in dataset.take(5):
    print(line.numpy())

#下边的函数定义了,标准化数据的函数,仅标准化一行
#使用map应用到整个数据集
n_inputs = 8 # X_train.shape[-1]
@tf.function
def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y

#这里其实整理了上边大部分的操作,取数,shuffle,标准化,batch(批处理？),prefetch

#prefetch是增高效率的关键使用了一种.使用了一种间隔处理方法,具体内容可看书
def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

tf.random.set_seed(42)

train_set = csv_reader_dataset(train_filepaths, batch_size=3)
for X_batch, y_batch in train_set.take(2):
    print("X =", X_batch)
    print("y =", y_batch)
    print()

train_set = csv_reader_dataset(train_filepaths, repeat=None)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

#与keras融合做一个神经网络模型,下面的代码先记下来以后学习下每个步骤
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

batch_size = 32
model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10,
          validation_data=valid_set)
model.evaluate(test_set, steps=len(X_test) // batch_size) #计算准确度

new_set = test_set.map(lambda X, y: X) # we could instead just pass test_set, Keras would ignore the labels
X_new = X_test
model.predict(new_set, steps=len(X_new) // batch_size)
