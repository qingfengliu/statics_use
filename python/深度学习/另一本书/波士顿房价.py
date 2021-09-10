# 波士顿房价数据。数据点较少,只有506个,分为404个训练样本和102个测试样本
# 取值范围不同有的是0~1,有的是1~12,还有范围是0~100
from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

