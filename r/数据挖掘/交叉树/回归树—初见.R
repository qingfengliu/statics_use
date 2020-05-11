
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
## rpart.control属于前剪枝 
Ctl<-rpart.control(minsplit=2,maxcompete=4,xval=10,maxdepth=10,cp=0)
set.seed(12345)

## kyphosis是rpart这个包自带的数据集  
## method：树的末端数据类型选择相应的变量分割方法:  
## 连续性method=“anova”,离散型method=“class”,计数型method=“poisson”,生存分析型method=“exp”  
## parms用来设置三个参数:先验概率、损失矩阵、分类纯度的度量方法（gini和information）  
## controlz指定修剪方法
#Purchase~. 可以理解为输入变量，这里表示除了Purchase的所有变量
#另一种形式y1~y2+y3
#输入变量~输出变量
TreeFit1<-rpart(Purchase~.,data=BuyOrNot,method="class",parms=list(split="gini"),control=Ctl)

#tree	画图所用的树模型。
#type	可取1,2,3,4.控制图形中节点的形式。
#fallen.leaves	fallen.leaves
#branch	控制图的外观。如branch=1，获得垂直树干的决策树。
#extra=2指出了辅助信息的显示方式
rpart.plot(TreeFit1,type=4,branch=0,extra=2)
printcp(TreeFit1)
plotcp(TreeFit1)


#prune指定CP值的后剪枝
TreeFit3<-prune(TreeFit1,cp=0.01) 
rpart.plot(TreeFit3,type=4,branch=0,extra=2)
printcp(TreeFit3)
plotcp(TreeFit3)