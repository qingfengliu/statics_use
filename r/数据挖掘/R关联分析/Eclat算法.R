#############eclat算法
library("arules")
library("arulesViz")
MyTrans<-read.transactions(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\事务原始数据.txt",format="basket",sep=",")
#echart(data=transactions对象名,parameter=NULL)
#parameter和aprior一样
MyFSets<-eclat(data=MyTrans,parameter=list(support=0.5,target="maximally frequent itemsets"))
inspect(MyFSets)
MyFSets<-eclat(data=MyTrans,parameter=list(support=0.5,target="frequent itemsets"))
plot(MyFSets)
#过滤出置信度大于0.6
MyRules<-ruleInduction(x=MyFSets,transactions=MyTrans,confidence=0.6)
inspect(sort(x=MyRules,by="lift"))

#############简单关联规则应用：发现连带销售商品
library("arules")
library("arulesViz")
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\购物篮数据.txt",header=TRUE,sep=",")
Data<-as.matrix(Data[,-1:-7])
MyTrans<-as(Data,"transactions")
summary(MyTrans)
MyRules<-apriori(data=MyTrans,parameter=list(support=0.1,confidence=0.5,target="rules"))
plot(MyRules,method="graph",control=list(arrowSize=2,main="连带销售商品可视化结果"))

#############简单关联规则应用：性别和年龄的啤酒选择性倾向对比
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\购物篮数据.txt",header=TRUE,sep=",")
Data<-Data[,c(4,7,14)]
Data$beer<-factor(Data$beer)
#给年龄分类
Data[,2]<-sapply(Data[,2],FUN=function(x){
  if(x %in% 0:29) x<-1 else
    if(x %in% 30:49) x<-2 else
      if(x %in% 50:59) x<-3})
Data$age<-factor(Data$age)
MyTrans<-as(Data,"transactions")
MyRules<-apriori(data=MyTrans,parameter=list(support=0.01,confidence=0.2,minlen=2,target="rules"),
                 appearance=list(rhs=c("beer=1"),
                                 lhs=c("age=1","age=2","age=3","sex=M","sex=F"),
                                 default="none"))
inspect(MyRules)
#is.subse判断是否有子集。(竖线代表true)
(SuperSetF<-is.subset(MyRules,MyRules))     
inspect(MyRules[-which(colSums(SuperSetF)>1)]) #浏览非冗余规则  
MyRules<-subset(x=MyRules,subset=quality(MyRules)$lift>1)
plot(MyRules,method="graph",control=list(arrowSize=2,main="性别与年龄的啤酒选择倾向对比"))
