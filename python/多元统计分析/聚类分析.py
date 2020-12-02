from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from skfuzzy.cluster import cmeans

###1.系统聚类
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

data=pd.read_csv(r"D:\书籍资料整理\多元统计分析\表3-5.csv")
temp=data[data.city.isin(['辽宁','浙江','河南','甘肃','青海'])]
temp=temp[['x1','x2','x3','x4','x5','x6','x7','x8']]

X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 4], [4, 0]])
#linkage:
#single 表示最短距离法
#Ward   最小化簇内平方和
#Average 重心法
#complete 最大距离法

#distance_threshold  距离阈值
#n_clusters 分类个数
clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage='single').fit(temp)
#分为两类的话与书中的一致
clustering.labels_

#绘制聚类谱图,绘制聚类谱图必须指定distance_threshold
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(clustering, truncate_mode='level')
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#2.模糊聚类
data=pd.read_csv(r"D:\书籍资料整理\多元统计分析\表3-7.csv")
train=data[['人均国内生产总值','粗死亡率','粗出生率','城镇人口比重','平均预期寿命','65岁及以上人口比重']].apply(
    lambda x: (x - np.mean(x)) / (np.std(x)))

train =np.asarray(train)
# train=preprocessing(train)
train=train.T

center, u, u0, d, jm, p, fpc = cmeans(train, m=2, c=3, error=0.0001, maxiter=1000)
#center:聚类的中心
#u是最后的的隶属度矩阵
#u0是初始化的隶属度矩阵
#d是最终的每个数据点到各个中心的欧式距离矩阵。
#jm是目标函数优化的历史。
#p是迭代的次数。
#fpc全称是fuzzy partition coefficient，是一个评价分类好坏的指标。它的范围是0到1，1是效果最好。后面可以通过它来选择聚类的个数。
result=u.T
result=pd.DataFrame(result,columns=['[,1]','[,2]','[,3]'])
result['country']=data['国家和地区']
#书中使用的是R的fanny函数,这里的结论与那个有些不同。
