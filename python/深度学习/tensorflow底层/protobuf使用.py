import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

#这步应该是定义一个文件体
person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }))

#序列化并且写入到tfrecord文件中.通常需要对csv的每行编写脚本进行序列化
with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    f.write(person_example.SerializeToString())

#加载example,首先定义每行内容
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),  #可变长数组
}
#固定长度特征被解析为规则张量,可变长度特征被解析为稀疏张量.可以使用tf.sparse.to_dense()转换为密集张量

for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
    parsed_example = tf.io.parse_single_example(serialized_example,
                                                feature_description)
tf.sparse.to_dense(parsed_example["emails"], default_value=b"")
parsed_example["emails"].values

from sklearn.datasets import load_sample_images

img = load_sample_images()["images"][0]
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.show()

data = tf.io.encode_jpeg(img)
example_with_image = Example(features=Features(feature={
    "image": Feature(bytes_list=BytesList(value=[data.numpy()]))}))

serialized_example = example_with_image.SerializeToString()
# then save to TFRecord. 这里保存的步骤貌似被省略了.

feature_description = { "image": tf.io.VarLenFeature(tf.string) }
example_with_image = tf.io.parse_single_example(serialized_example, feature_description)
decoded_img = tf.io.decode_jpeg(example_with_image["image"].values[0])
decoded_img = tf.io.decode_image(example_with_image["image"].values[0])   #可以




#暂时就到这里了,如果需要用到的时候可以深究.

#