import pandas as pd
import os

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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

#分层抽样,为了训练集贴近  实际情况
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


#首先备份数据

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


#虽然这个方法一个函数就能实现 但是sklearn 提供了标准的一套法则,无论难易都遵循这个法则
imputer = SimpleImputer(strategy="median")
#需要删除了属性类型才能使用sklearn 学习
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

#2.转换流水线


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

#上边这些主要是为了还原上一步的清洗数据.但是可能并不完全与书中一致,因为缺少了一步.

#使用线性回归,


lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 拿一些测试集中的数据来测试一下.可以发现不是很准确
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
test_pre=lin_reg.predict(some_data_prepared)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
print(type(some_data_prepared),type(test_pre))

exit()
#然后计算整个测试集上回归模型的均方误差

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#RMSE可以这样直接求出,但是和mean_squared_error在开根号求得的值并不一样.肯定是经过什么优化的结果
housing_predictions = lin_reg.predict(housing_prepared)
lin_mae = mean_absolute_error(housing_labels, housing_predictions)

print(lin_mae)   #可以说比较差强人意.是由于欠拟合造成的.

#使用决策树
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_absolute_error(housing_labels, housing_predictions)
print(tree_mse)

#结果显示为0,则可能存在过拟合.

#这里要选择一个与原来理解不同的处理方式,将测试集进行再分割用来训练和测试
#这里将测试集做成10折交叉验证数据集
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

#其实这个模型的结果也不咋地，甚至其实更糟.可在与线性模型去比较可以的出,还不如线性模型,所以存在过拟合.
#下边使用随机森林
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print('forest rmse:',forest_rmse)

#交叉验证
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)

#这里随机森林会好些,但是在分离的集合上仍存在分数较高的情况.显示仍存在一些过拟合

#下边是调参过程 ，可以使用GridSearchCV类只要把要调的参数告诉他,并且把要尝试的值告诉他即可

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
#数字中两个位置的数据都会尝试所以光是参数就会尝试12+6=18中组合

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_  #得到最好的参数
grid_search.best_estimator_ #甚至可以得到最好的估计器

# 当然也可以获得RMSE结果
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#另一种随机调参的方式为RandomizedSearchCV 可以用于 需要调整很多随机参数时。
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

#通过测试集评估系统,
#首先点估计

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

#其次可以求泛化误差的95%置信区间,服从t分布
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

#也可以使用z得分
mean = squared_errors.mean()
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)

