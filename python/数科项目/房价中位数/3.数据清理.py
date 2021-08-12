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
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
housing["income_cat"].hist()  #频数分布直方图  必须使用plt.show才能显示
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
#1.接上一张随机抽样之后.我们需要清洗数据.
#首先需要处理,空值,基本上来说有三种选项.(1)放弃这些相应的区域.(2)放弃整个属性.(3)将缺失的值设置为某个值(0、平均数或中位数)
#通过dropna()、drop()和fillna()

#首先备份数据
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

#三种方式择其一
# strat_train_set.dropna(subset=["total_bedrooms"])    # option 1
# strat_train_set.drop("total_bedrooms", axis=1)       # option 2

# median = housing["total_bedrooms"].median()
# strat_train_set["total_bedrooms"].fillna(median, inplace=True) # option 3  在选用此方法需要记录一下，填充的值
#sklearn 提供了处理填充问题的函数.
from sklearn.impute import SimpleImputer
#虽然这个方法一个函数就能实现 但是sklearn 提供了标准的一套法则,无论难易都遵循这个法则
imputer = SimpleImputer(strategy="median")
#需要删除了属性类型才能使用sklearn 学习
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
#这里imputer仅求了中位数,在imputer的statistics_中存储中位数

X = imputer.transform(housing_num)
#然后这里才求出转换出的结果,然后转换c成pandas的DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)

#处理文本属性
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

#文本转换为数字.使用sklearn的ordinalEncoder类
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
ordinal_encoder.categories_  #可以查看各类别顺序

# 这样处理有个问题,机器学习算法会认为两个相近的值比两个离得较短的值更为相似一些
# 所以可以把他们变成哑变量,使用onehotencoder

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray()

#2.转换流水线
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# 转换流水线,Pipeline构造函数会通过一系列名称/估算器的配对来定义步骤序列.除了最后一个是
# 估算器之外,前面都必须是转换器(也就是必须有fit_transform()方法)
# 调用流水线的fit()方法时,会依次调用fit_transform(),将一个调用的输出作为参数传递给下一个
# 调用方法,直到传递到最终的估算器,则会调用fit()方法
# 流水线的方法与最终的估算器的方法相同.由于上例最终StandardScaler为转换器,因此可用fit_transform方法

# 有一个更方便的流水线函数.  元组由三个参数构建 第一个为转换器的名字(字符串随意起名),第二个为转换器
# 第三个为转换器能用到的列名
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)




