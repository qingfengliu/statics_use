import pandas as pd
# from dataprep.eda import plot
import warnings
from scorecardbundle.feature_discretization import ChiMerge as cm
from scorecardbundle.feature_discretization import FeatureIntervalAdjustment as fia
from scorecardbundle.feature_encoding import WOE as woe
from scorecardbundle.feature_selection import FeatureSelection as fs
from scorecardbundle.model_training import LogisticRegressionScoreCard as lrsc
from scorecardbundle.model_evaluation import ModelEvaluation as me
from sklearn.model_selection import train_test_split


train_data = pd.read_csv(r'D:\书籍资料整理\工作接触的模型\GiveMeSomeCredit\cs-training.csv', index_col=0)
train_data.columns = ['严重违约', '可用额度比例','年龄', '35-69天逾期次数', '负债比例','月收入','普通贷款数量','高于90天逾期次数','不动产贷款数量','60-89天逾期次数','家属数量']
train_data = train_data[['年龄','家属数量','月收入','负债比例','可用额度比例','普通贷款数量','不动产贷款数量','35-69天逾期次数','60-89天逾期次数','高于90天逾期次数','严重违约']]

# 手工探索数据
print(train_data.shape)
print(train_data.isnull().sum()/train_data.shape[0])
print(train_data.describe().T)
print(train_data['严重违约'].value_counts())  #可以看到不均衡
print(train_data['严重违约'].sum()/train_data['严重违约'].count()) #0.06684
# 使用EDA工具探索数据
# plot(train_data)

#先说结论,模型是一个不平衡数据集,并且月收入缺失较严重,家属数量缺失一般.
#示例代码使用了随机森林填充法,这里可以试验下与0填充对比
#随机森林填充法、假设要填充变量与其他值之间有关系
from sklearn.ensemble import RandomForestRegressor


def fill_income_missing(data, to_fill):
    df = data.copy()
    columns = [*df.columns]
    columns.remove(to_fill)

    # 移除有缺失值的列
    columns.remove('家属数量')
    X = df.loc[:, columns]
    y = df.loc[:, to_fill]
    X_train = X.loc[df[to_fill].notnull()]
    y_train = y.loc[df[to_fill].notnull()]
    X_pred = X.loc[df[to_fill].isnull()]
    rfr = RandomForestRegressor(random_state=22, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_pred).round()
    df.loc[df[to_fill].isnull(), to_fill] = y_pred
    return df


def fill_dependents_missing(data, to_fill):
    df = data.copy()
    columns = [*df.columns]
    columns.remove(to_fill)

    X = df.loc[:, columns]
    y = df.loc[:, to_fill]
    X_train = X.loc[df[to_fill].notnull()]
    y_train = y.loc[df[to_fill].notnull()]
    X_pred = X.loc[df[to_fill].isnull()]
    rfr = RandomForestRegressor(random_state=22, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_pred).round()
    df.loc[df[to_fill].isnull(), to_fill] = y_pred
    return df

train_data = fill_income_missing(train_data, '月收入')
train_data = fill_dependents_missing(train_data, '家属数量')
print(train_data.isnull().sum())
#删除年龄小于0的数据
train_data = train_data.loc[train_data['年龄'] > 0]
#这里是去除预期数据中的异常值(逾期次数较多的)
columns = ['35-69天逾期次数','60-89天逾期次数','高于90天逾期次数']
train_data.loc[:, columns].plot.box(vert=False)
train_data = train_data[(train_data['35-69天逾期次数'] < 90) & (train_data['60-89天逾期次数'] < 90)  & (train_data['高于90天逾期次数'] < 90)]
print('-------------------------数据探索完------------------------')

X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
trans_cm = cm.ChiMerge(max_intervals=10, min_intervals=5, output_dataframe=True)  #卡分分箱
result_cm = trans_cm.fit_transform(X_train, y_train)
print('特征切分',trans_cm.boundaries_) # 每个特征的区间切分

trans_woe = woe.WOE_Encoder(output_dataframe=True)
result_woe = trans_woe.fit_transform(result_cm, y_train)
print('每个特征的信息值',trans_woe.iv_) # 每个特征的信息值 (iv)
print('每个特征的WOE字典和信息值',trans_woe.result_dict_) # 每个特征的WOE字典和信息值 (iv)

#手动调整分箱:观察每一个特征的分布和响应率
col = '年龄'
fia.plot_event_dist(result_cm[col],y_train,x_rotation=60)
new_x = cm.assign_interval_str(X_train[col].values,[22, 33, 43, 53, 62, 67, 74]) # apply new interval boundaries to the feature
woe.woe_vector(new_x, y_train.values) # check the information value of the resulted feature that applied the new intervals
fia.plot_event_dist(new_x,y_train, x_label=col,x_rotation=60)
feature_list = []
result_cm[col] = new_x # great explainability and predictability. Select.
feature_list.append(col)
print(feature_list)

#进行WOE编码
trans_woe = woe.WOE_Encoder(output_dataframe=True)
result_woe = trans_woe.fit_transform(result_cm[feature_list], y_train)
print(result_woe.head())
print(trans_woe.iv_)

#剔除预测力过低（通常用IV不足0.02筛选）、以及相关性过高引起共线性问题的特征。
# (相关性过高的阈值默认为皮尔森相关性系数大于0.6，可通过threshold_corr参数调整)
fs.selection_with_iv_corr(trans_woe, result_woe) # corr_with 列示了与该特征相关性过高的特征和相关系数

#模型训练
model = lrsc.LogisticRegressionScoreCard(trans_woe, PDO=-20, basePoints=100, verbose=True)
model.fit(result_woe, y_train)
print(model.woe_df_) # 从woe_df_属性中可得评分卡规则


sc_table = model.woe_df_.copy()
result = model.predict(X_train[feature_list],
                       load_scorecard=sc_table)  # Scorecard should be applied on the original feature values
result_test = model.predict(X_test[feature_list],
                            load_scorecard=sc_table)  # Scorecard should be applied on the original feature values
result.head()  # if model object's verbose parameter is set to False, predict will only return Total scores
# Train
evaluation = me.BinaryTargets(y_train, result['TotalScore'])
evaluation.plot_all()

# Validation
evaluation = me.BinaryTargets(y_test, result_test['TotalScore'])
evaluation.plot_all()