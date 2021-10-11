import pandas as pd
import scipy.sparse as sparse
import numpy as np
import random
import implicit
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import random

random.seed(42)
# https://blog.csdn.net/weixin_42608414/article/details/90319447
# 使用派神的文章
# 版权声明：本文为CSDN博主「-派神 -」的原创文章，遵循CC
# 4.0
# BY - SA版权协议，转载请附上原文出处链接及本声明。

articles_df = pd.read_csv(r'D:\书籍资料整理\kaggle\CI&T\shared_articles.csv')
interactions_df = pd.read_csv(r'D:\书籍资料整理\kaggle\CI&T\users_interactions.csv')
articles_df.drop(['authorUserAgent', 'authorRegion', 'authorCountry'], axis=1, inplace=True)
interactions_df.drop(['userAgent', 'userRegion', 'userCountry'], axis=1, inplace=True)
# print(articles_df.head())
# print(interactions_df.head())

#
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.drop('eventType', axis=1, inplace=True)
df = pd.merge(interactions_df[['contentId','personId', 'eventType']], articles_df[['contentId', 'title']], how = 'inner', on = 'contentId')
# print(df.head())
# print(df['eventType'].value_counts())

# VIEW: 查看，表示用户点击过该篇文章
# LIKE:  点赞，表示用户喜好该篇文章。
# BOOKMARK: 加入书签，用户已将该文章加入书签，以便将来查看。这是用户对该文章表现出感兴趣的迹象
# COMMENT CREATED: 创建评论，说明用户被这篇文章所吸引，同时希望表达自己的观点。
# FOLLOW: 关注，用户关注了文章的作者，这应该说明用户对这位作者的文章非常感兴趣
# ————

event_type_strength = {
    'VIEW': 1.0,
    'LIKE': 2.0,
    'BOOKMARK': 3.0,
    'FOLLOW': 4.0,
    'COMMENT CREATED': 5.0,
}

df['eventStrength'] = df['eventType'].apply(lambda x: event_type_strength[x])

#删除了重复值,说明重复的动作是无效的特别是对浏览行为
df = df.drop_duplicates()
grouped_df = df.groupby(['personId', 'contentId', 'title']).sum().reset_index()
grouped_df.sample(10)

#把ID转换成短整形int16
grouped_df['title'] = grouped_df['title'].astype("category")
grouped_df['personId'] = grouped_df['personId'].astype("category")
grouped_df['contentId'] = grouped_df['contentId'].astype("category")
grouped_df['person_id'] = grouped_df['personId'].cat.codes
grouped_df['content_id'] = grouped_df['contentId'].cat.codes
grouped_df.sample(10)

# 在矩阵分解（matrix factorization）中使用的一种算法。有一个稀疏矩阵，假设这个矩阵是低阶的，可以分解成两个小矩阵相乘。
# 然后交替对两个小矩阵使用最小二乘法，算出这两个小矩阵，就可以估算出稀疏矩阵缺失的值：
# 这里估算出系数矩阵的缺失值是我们最终的目的就是估算用户对没看过的文章的感兴趣程度.

#在这里我们将使用implicit库中的als.AlternatingLeastSquares方法来实现交替最小二乘法。
# 不过，这里我们首先要创建两个稀疏矩阵:内容-人-评分矩阵，人-内容-评分矩阵。


sparse_content_person = sparse.csr_matrix(
    (grouped_df['eventStrength'].astype(float), (grouped_df['content_id'], grouped_df['person_id'])))
sparse_person_content = sparse.csr_matrix(
    (grouped_df['eventStrength'].astype(float), (grouped_df['person_id'], grouped_df['content_id'])))

print(sparse_content_person.shape)
print(sparse_person_content.shape)
# 我们看到文章总数为2979，用户总数为1895
# 内容-人-评分矩阵(sparse_content_person)的每一行代表一篇文章，每一列代表一个用户。
# 人-内容-评分矩阵(sparse_person_content)的每一行代表一个用户，每一列代表一篇文章。


alpha = 15
data = (sparse_content_person * alpha).astype('double')

model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
model.fit(data)
# 一、推荐内容相似的文章
# 模型训练完成以后，我们可以计算文章之间的相似度，并根据相似度来进行推荐:我们以content_id = 235的文章为例,
# 这篇文章的标题是"Artificial intelligence is hard to see"，这是一篇有关人工智能的文章，
# 我们的目标是在所有的文章的标题中找出和它最相似的10篇文章标题。这里我们主要通过计算文章之间的相似度，在计算相似度的过程中有以下几个步骤:
# 从模型中获取人和内容的向量
# 计算内容的向量的范数
# 计算相似度得分
# 获取相似度得分最大的10篇文章
# 生成这10篇文章的content_id和title的元组


# 获取用户矩阵
content_id=235
n_similar=10

person_vecs = model.user_factors
# 获取内容矩阵
content_vecs = model.item_factors
# 计算内容的向量的范数
content_norms = np.sqrt((content_vecs * content_vecs).sum(axis=1))
# 计算指定的content_id 与其他所有文章的相似度
scores = content_vecs.dot(content_vecs[content_id]) / content_norms
# 获取相似度最大的10篇文章
top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
# 组成content_id和title的元组
similar = sorted(zip(top_idx, scores[top_idx] / content_norms[content_id]), key=lambda x: -x[1])

#我们注意到我们的用户矩阵和我们的内容矩阵都只有20列，
# 这是因为我们在定义模型时设置了factors=20。
# 这里的用户矩阵和内容矩阵就类似与前面交替最小二乘法示意图中介绍的User Feature Matrix和Movie Feature Matrix.

print(person_vecs.shape)
print(content_vecs.shape)

#下面我们展示这10篇最相似的文章title:

# for content in similar:
#     idx, score = content
#     print(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0],"|",score)

#这些文章似乎都和人工智能有关。其中第一篇文章就是content_id = 235的文章本身，
# 正如我们前面说content_id = 235要和所有文章计算相似度，这当然也包含它自己本身，
# #所以content_id = 235的文章和自己的相似度应该是最大的，因此应该排第一位，其余9篇文章标题则是按照相似度由高到低排列的。

#二、为用户推荐他可能感兴趣的文章
#接下来我们要为用户推荐他们没有看过的(即没有发生过交互行为),但是他们可能会敢兴趣的文章。我们首先要定义一个推荐函数，在推荐函数中我们主要要做以下3件事:

#将指定用户的用户向量乘以内容矩阵，得到该用户对所有文章的评分向量
#从评分向量中过滤掉用户已经评分过的文章(将其评分值置为0),因为用户已经发生过交互行为的文章不应该被推荐
#将余下的评分分数排序，并输出分数最大的10篇文章。
#貌似这个函数已经被纳入到implicit包中了

def recommend(person_id, sparse_person_content, person_vecs, content_vecs, num_contents=10):
    # *****************得到指定用户对所有文章的评分向量******************************
    # 将该用户向量乘以内容矩阵(做点积),得到该用户对所有文章的评价分数向量
    rec_vector = person_vecs[person_id, :].dot(content_vecs.T).toarray()

    # **********过滤掉用户已经评分过的文章(将其评分值置为0),因为用户已经发生过交互行为的文章不应该被推荐*******
    # 从稀疏矩阵sparse_person_content中获取指定用户对所有文章的评价分数
    person_interactions = sparse_person_content[person_id, :].toarray()
    # 为该用户的对所有文章的评价分数+1，那些没有被该用户看过(view)的文章的分数就会等于1(原来是0)
    person_interactions = person_interactions.reshape(-1) + 1
    # 将那些已经被该用户看过的文章的分数置为0
    person_interactions[person_interactions > 1] = 0
    # 将该用户的评分向量做标准化处理,将其值缩放到0到1之间。
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
    # 过滤掉和该用户已经交互过的文章，这些文章的评分会和0相乘。
    recommend_vector = person_interactions * rec_vector_scaled

    # *************将余下的评分分数排序，并输出分数最大的10篇文章************************
    # 根据评分值进行排序,并获取指定数量的评分值最高的文章
    content_idx = np.argsort(recommend_vector)[::-1][:num_contents]

    # 定义两个list用于存储文章的title和推荐分数。
    titles = []
    scores = []

    for idx in content_idx:
        # 将title和分数添加到list中
        titles.append(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'title': titles, 'score': scores})

    return recommendations

# 在上面的函数中，我们首先从稀疏矩阵sparse_person_content中获取指定用户对所有文章的评分，
# 这里需要注意一点的是，如果某个用户对某篇文章没有发生过交互行为(VIEW,LIKE,BOOKMARK,COMMENT CREATED,FOLLOW),
# 那么在原始的数据集中是不存在这条交互记录的，但是当我们使用了稀疏矩阵的.toarray()方法后，
# 那些用户没有发生过交互的所有文章都会被展示出来，只不过对那些文章的评分值都会被置为0，
# 因此toarray()方法展现的是所有用户对所有文章的评分结果。


# 从model中获取经过训练的用户和内容矩阵,并将它们存储为稀疏矩阵
person_vecs = sparse.csr_matrix(model.user_factors)
content_vecs = sparse.csr_matrix(model.item_factors)

# 为指定用户推荐文章。
person_id = 50
recommendations = recommend(person_id, sparse_person_content, person_vecs, content_vecs)
print(recommendations)

# 三、评估
# 大体上来说评估推荐系统的表现主要是通过计算推荐结果的“命中率”来考察推荐算法表现，主要思想是这样的，
# 我们从现有的评分矩阵中分离出少部分评分数据(如20%左右),将剩余的80%的推荐数据用来训练推荐算法模型，
# 然后让推荐模型对用户未评分过的文章进行推荐，在推荐结果中我们考察其中是否包含了之前被分离出来的那20%的文章，
# 同时我们计算“命中率”或AUC作为评价指标。

# 现在我们要做的是从评分数据中创建训练集和测试集，在测试集中我们删除了20%的有过交互行为的评分数据，
# 在测试集中我们将所有的有过交互行为评分置为1，这样就测试集变成了一个二分类数据的集合。


import random


def make_train(ratings, pct_test=0.2):
    test_set = ratings.copy()  # 拷贝一份评分数据当作测试集
    test_set[test_set != 0] = 1  # 将有评分数据置为1，我们要模拟成二分类数据集

    training_set = ratings.copy()  # 拷贝一份评分数据当作训练集
    nonzero_inds = training_set.nonzero()  # 找到有过评分(有交互行为，评分数不为0)的数据的索引。
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))  # 将它们组成元组并存放在list中

    random.seed(0)  # 设置随机数种子

    num_samples = int(np.ceil(pct_test * len(nonzero_pairs)))  # 获取20%的非0评价的数量
    samples = random.sample(nonzero_pairs, num_samples)  # 随机从非零评价的索引对中抽样20%

    content_inds = [index[0] for index in samples]  # 从样本中得到文章列(第一列)索引值
    person_inds = [index[1] for index in samples]  # 从样本中得到文章列(第二列)索引值

    training_set[content_inds, person_inds] = 0  # 在训练集中将这20%的随机样本的评分值置为0
    training_set.eliminate_zeros()  # 在测试集中删除这0元素

    return training_set, test_set, list(set(person_inds))


content_train, content_test, content_persons_altered = make_train(sparse_content_person, pct_test=0.2)

#计算AUC分数
def auc_score(predictions, actual):
    fpr, tpr, thresholds = metrics.roc_curve(actual, predictions)
    return metrics.auc(fpr, tpr)


# 计算评价AUC分数
def calc_mean_auc(training_set, altered_persons, predictions, test_set):
    store_auc = []  # 用来存储那些在训练集中被删除评分的用户的AUC
    popularity_auc = []  # 用来存储最受欢迎的文章的AUC
    pop_contents = np.array(test_set.sum(axis=1)).reshape(-1)  # 在测试集中按列合计所有评价分数，以便找出最受欢迎的文章。
    content_vecs = predictions[1]
    for person in altered_persons:  # 迭代那些在训练集中被删除评分的那20%的用户
        training_column = training_set[:, person].toarray().reshape(-1)  # 在训练集中找到对应用户的那一列
        zero_inds = np.where(training_column == 0)  # 找出所有没有发生过交互行为的评分的索引,这其中也包括被删除评分的索引

        # 对用户没有交互过的文章预测用户对它们的评分
        person_vec = predictions[0][person, :]
        pred = person_vec.dot(content_vecs).toarray()[0, zero_inds].reshape(-1)

        # 获取预测的评分，预测评分包含用户交互过的文章的评分(原评分为0)和那20%被强制置为0的实际评分
        actual = test_set[:, person].toarray()[zero_inds, 0].reshape(-1)

        # 从所有文章评价总和中过滤出过滤出那么没有评价过的文章的合计总分(每篇文章各自的合计总分)
        pop = pop_contents[zero_inds]

        store_auc.append(auc_score(pred, actual))  # 计算当前用户的预测和实际评分的AUC

        popularity_auc.append(auc_score(pop, actual))  # 计算合计总分和实际评分的AUC

    return float('%.3f' % np.mean(store_auc)), float('%.3f' % np.mean(popularity_auc))


print(calc_mean_auc(content_train, content_persons_altered,
              [person_vecs, content_vecs.T], content_test))

#https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101/notebook
#另外注这里边实现了一个recsys推荐系统大会中的算法.具体详情不清楚

