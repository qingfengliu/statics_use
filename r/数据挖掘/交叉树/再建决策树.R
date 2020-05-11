library("rpart")
library("rpart.plot")

BuyOrNot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\消费决策数据.txt",header=TRUE)
BuyOrNot$Income<-as.factor(BuyOrNot$Income)  #指定收入为因子
BuyOrNot$Gender<-as.factor(BuyOrNot$Gender)  #指定性别为因子


set.seed(12345)
(TreeFit2<-rpart(Purchase~.,data=BuyOrNot,method="class",parms=list(split="gini"))) 
rpart.plot(TreeFit2,type=4,branch=0,extra=2)
printcp(TreeFit2)

#按系统默认值采取CP初始值0.01
#使用rpart建模后的结果
#1) root 431 162 0 (0.6241299 0.3758701)  
#2) Income=1,2 276  88 0 (0.6811594 0.3188406) *
#  3) Income=3 155  74 0 (0.5225806 0.4774194)  
#  6) Age< 44.5 128  56 0 (0.5625000 0.4375000) *
#  7) Age>=44.5 27   9 1 (0.3333333 0.6666667) *
#在2)节点后有个*代表子叶结点。income=1，2 错判率为0.32 置信度为0.68
#可在rpart.plot看出分类树整个的图
#printcp
