import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import KMeans

blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)
plt.figure(figsize=(8, 4))
plot_clusters(X)
plt.show()

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)  #fit因为是聚类所以与上边的学习不同

#类中心
print(kmeans.cluster_centers_)

print(kmeans.labels_)

#预测新值
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

#Voronoi图
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
plt.show()

#返回到各个中心点的距离
kmeans.transform(X_new)

#最坏的情况下复杂度会随着实例数量的增加呈指数增加.
#实际上这种情况很少发生.并且k-means通常是最快的聚类算法之一.


#中心点初始化
good_init=np.array([[-3,3],[-3,2],[-3,1],[-1,2],[0,2]])
kmeans=KMeans(n_clusters=5,init=good_init,n_init=1)
kmeans.fit_predict(X)
#参数n_init 指算法执行次数,sklearn会选择最好的那个,因为k-means法会陷入局部最小,通过多次
#初始化中心点来缓解这个问题。k-means通过惯性来衡量这个是否是最好
#这种固定中心情况下貌似没有inertia_这个指标

print(kmeans.inertia_)
#另一种是使用score这个数是惯性的相反数,因为sklearn秉承着越大越好的原则
print(kmeans.score(X))

#sklearn使用k-means++算法来初始化点,使得算法收敛到次优解的可能性很小.
#并且在迭代步骤中也进行了优化,使用了三角不等式,并通过追踪实例和中心点
#之间的上下限距离来实现这一目的.

#小批量k-means法，速度快相应的 惯性会高一些
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)

#寻找最佳分类,轮廓数
from sklearn.metrics import silhouette_score
print('轮廓数:',silhouette_score(X, kmeans.labels_))
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]
plt.figure(figsize=(8, 3))
# plt.title('轮廓数')
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.55, 0.7])
plt.show()

#k-means局限,当集群具有不同的大小,不同的密度或非球形时效果不佳.
#另外k-means必须特征缩放否则效果不好



