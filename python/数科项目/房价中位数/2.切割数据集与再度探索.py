#这步切割数据集.主要是为了将数据集分为测试集和训练集.
#但是这一步的原则是,训练集应该尽量固定,不受测试运行的影响
#这里有两种方式1.随机抽取并且设置随机数种子相同那么产生的随机数就是一致的.
#对于这种方式,其实书中的函数与sklearn包里的函数train_test_split
#那么就不记录了.这里纯随机的缺点是当拓展数据后就无法固定了
#第二种方式为,给每一行一个唯一标识符.然后计算一个哈希值然后这个
#取最大哈希值<20%.
#2.分层抽样,分层抽样更能体现总体的结构减小误差.
#3.书中的例子逻辑为,收入中位数,体现房价.
#那么将收入中位数分层,然后进行分层抽样

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from zlib import crc32
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#随机分
pd.set_option('display.max_columns', None)
# fetch_housing_data()   #获取数据并保存在HOUSING_PATH中
housing = load_housing_data()
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)


#唯一值,
#当使用行号时要保证更新数据源的时候不会删除任何行.否则需要找个更稳定的做唯一标志.

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

#分层抽样
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
housing["income_cat"].hist()  #频数分布直方图  必须使用plt.show才能显示
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#这里看到测试集与全集的比例就正好了
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(housing["income_cat"].value_counts() / len(housing))

#可视化洞见
housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude")
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)  #alpha透明度
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
#不仅需要透明度,可以定义散点图.s定义散点图圆的半径使用人口.颜色,选项为c使用房屋价值中位数？然后用参数cmap预定义颜色.
#
plt.legend()
# plt.show()
# save_fig("bad_visualization_plot")  #这里就不做图片保存了.
# images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
# os.makedirs(images_path, exist_ok=True)
# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
# filename = "california.png"
# print("Downloading", filename)
# url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
# urllib.request.urlretrieve(url, os.path.join(images_path, filename))
# import matplotlib.image as mpimg
# california_img=mpimg.imread(os.path.join(images_path, filename))
# ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
#                   s=housing['population']/100, label="Population",
#                   c="median_house_value", cmap=plt.get_cmap("jet"),
#                   colorbar=False, alpha=0.4)
# plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
#            cmap=plt.get_cmap("jet"))
# plt.ylabel("Latitude", fontsize=14)
# plt.xlabel("Longitude", fontsize=14)
#
# prices = housing["median_house_value"]
# tick_values = np.linspace(prices.min(), prices.max(), 11)
# cbar = plt.colorbar(ticks=tick_values/prices.max())
# cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
# cbar.set_label('Median House Value', fontsize=16)
#
# plt.legend(fontsize=16)
# save_fig("california_housing_prices_plot")


#2.寻找相关性
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))



attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
#scatter_matrix,相关性图.

#可以合并各别属性当作综合属性.

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
#综合属性,