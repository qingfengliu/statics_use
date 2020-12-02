import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,KernelPCA,SparsePCA

#KernelPCA 核主成分分析
#PCA 稀疏主成分分析
#SparsePCA 稀疏主成分分析

data=pd.read_csv(r"D:\书籍资料整理\多元统计分析\例5-1.csv")
x=data[['农','制造业','电力','建筑业','住宿','金融','房地产','教育','文化']]
x=np.asarray(x)
pca = PCA(n_components=2)
X_reduced =pca.fit_transform(x)
#结果与书中一致
pca.explained_variance_ratio_

#这个应该是实际通过主成分转换后的结果
np.dot(x,pca.components_.T)
#这个是通过sklearn的PCA转换后的结果,其结果有些不一样.有资料说都是对的。
X_reduced
#components_ 主成分方程的系数
#n_components='mle'  PCA用最大似然估计
#n_components = 0.97 svd_solver =='full'  按信息量占比选超参数
#explained_variance_  每个分量所解释的方差量
#explained_variance_ratio_所选择的每个组成部分所解释的方差百分比
#mean_每个列的均值
