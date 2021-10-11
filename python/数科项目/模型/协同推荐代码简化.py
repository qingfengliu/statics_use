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

articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.drop('eventType', axis=1, inplace=True)
df = pd.merge(interactions_df[['contentId','personId', 'eventType']], articles_df[['contentId', 'title']], how = 'inner', on = 'contentId')

event_type_strength = {
    'VIEW': 1.0,
    'LIKE': 2.0,
    'BOOKMARK': 3.0,
    'FOLLOW': 4.0,
    'COMMENT CREATED': 5.0,
}

df['eventStrength'] = df['eventType'].apply(lambda x: event_type_strength[x])


df = df.drop_duplicates()
grouped_df = df.groupby(['personId', 'contentId', 'title']).sum().reset_index()
grouped_df.sample(10)


grouped_df['title'] = grouped_df['title'].astype("category")
grouped_df['personId'] = grouped_df['personId'].astype("category")
grouped_df['contentId'] = grouped_df['contentId'].astype("category")
grouped_df['person_id'] = grouped_df['personId'].cat.codes
grouped_df['content_id'] = grouped_df['contentId'].cat.codes
grouped_df.sample(10)



sparse_content_person = sparse.csr_matrix(
    (grouped_df['eventStrength'].astype(float), (grouped_df['content_id'], grouped_df['person_id'])))
sparse_person_content = sparse.csr_matrix(
    (grouped_df['eventStrength'].astype(float), (grouped_df['person_id'], grouped_df['content_id'])))

print(sparse_content_person.shape)
print(sparse_person_content.shape)


alpha = 15
data = (sparse_content_person * alpha).astype('double')

model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
model.fit(data)

item_id= 235

similar = model.similar_items(item_id, N=10)


for content in similar:
    idx, score = content
    print(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0],"|",score)



user_id = 50

recommendations = model.recommend(user_id, sparse_person_content)


for content in recommendations:
    idx, score = content
    print(grouped_df.title.loc[grouped_df.content_id == idx].iloc[0],"|",score)