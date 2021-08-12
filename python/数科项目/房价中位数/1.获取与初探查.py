# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
import pandas as pd
# Common imports


# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tarfile
import urllib.request

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
# 这里配置了图片保存的路径
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


#该地址已经无法访问
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#获取数据并保存在HOUSING_PATH中
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    print(housing_url+tgz_path)
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)   #解压过程,
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

pd.set_option('display.max_columns', None)
# fetch_housing_data()   #获取数据并保存在HOUSING_PATH中
housing = load_housing_data()
# print(housing.head())  #这一步主要查看数据类型和样子
# print(housing.info())   #进一步查看类型,确认除了最后一个都是float蕾西

# print(housing["ocean_proximity"].value_counts())  #这个函数可以看出属性,ocean_proximity到底有什么值,每个值有多少个.

# print(housing.describe())  #不建议用这个函数看统计指标,都是数字很不直观

#以下主要为了看到数据的分布,可见除了经纬度,人口外基本属性值都是右尾分布的。
#可通过一些转换,尽量使分布称钟形
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()



