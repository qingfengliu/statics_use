
library("rpart")
library("rpart.plot")

############分类回归树: rpart包
BuyOrNot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\消费决策数据.txt",header=TRUE)
BuyOrNot$Income<-as.factor(BuyOrNot$Income)  #指定收入为因子
BuyOrNot$Gender<-as.factor(BuyOrNot$Gender)  #指定性别为因子

## rpart.control对树进行一些设置  
## xval，交叉验证10为10着交叉验证  
## minsplit是最小分支节点数(样本)，这里指大于等于2，那么该节点会继续分划下去，否则停止  
## minbucket：叶子节点最小样本数  
## maxdepth：树的深度  
## rpart.control专门用于设置枝剪方式，指某个点的复杂度，对每一步拆分,模型的拟合优度必须提高的程度 
Ctl<-rpart.control(minsplit=2,maxcompete=4,xval=10,maxdepth=10,cp=0)
set.seed(12345)

## kyphosis是rpart这个包自带的数据集  
## method：树的末端数据类型选择相应的变量分割方法:  
## 连续性method=“anova”,离散型method=“class”,计数型method=“poisson”,生存分析型method=“exp”  
## parms用来设置三个参数:先验概率、损失矩阵、分类纯度的度量方法（gini和information）  
## control？
TreeFit1<-rpart(Purchase~.,data=BuyOrNot,method="class",parms=list(split="gini"),control=Ctl)
rpart.plot(TreeFit1,type=4,branch=0,extra=2)
printcp(TreeFit1)
plotcp(TreeFit1)