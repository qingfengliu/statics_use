library("neuralnet")
BuyOrNot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\消费决策数据.txt",header=TRUE)

##########neurealnet建立神经网络
set.seed(12345)

#neurealnet(输出变量~输入变量,data=数据框名,hidden=1,threshold=0.01,stepmax=100000,rep=迭代周期,
#err.fct=误差函数名,linear.output=TRUE,learningrate=学习率,algorithm=算法名)
#hidden=c(3,2,1),表示有3个隐层,第1~3个隐藏分别包含3,2,1个隐节点。
#参数threshold用于迭代停止条件。当权重的最大调整量小于指定值(默认值为0.01)时迭代终止。
#stepmax用于指定迭代停止条件。当迭代次数达到指定次数(默认为100000次)时迭代终止。

#根据是否购买,年龄、性别、收入建模

(BPnet1<-neuralnet(Purchase~Age+Gender+Income,data=BuyOrNot,hidden=2,err.fct="ce",linear.output=FALSE))
BPnet1$result.matrix
BPnet1$weight
BPnet1$startweights 