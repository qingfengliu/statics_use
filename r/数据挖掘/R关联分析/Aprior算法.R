library("arules")

###########生成transactoins对象：方式一
MyList<-list(   
  c("A","C","D"),
  c("B","C","E"),
  c("A","B","C","E"),
  c("B","E")
)
names(MyList)<-paste("Tr",c(1:4),sep="")
MyTrans<-as(MyList,"transactions")
summary(MyTrans)
inspect(MyTrans)
image(MyTrans)

###########生成transactoins对象：方式二
MyFact<-matrix(c(
  1,0,1,1,0,
  0,1,1,0,1,
  1,1,1,0,1,
  0,1,0,0,1
),nrow=4,ncol=5,byrow=TRUE)
dimnames(MyFact)<-list(paste("Tr",c(1:4), sep = ""),c("A","B","C","D","E"))
MyFact
(MyTrans<-as(MyFact,"transactions"))
(as(MyTrans,"data.frame"))

###########生成transactoins对象：方式三
MyT<-data.frame(
  TID=c(1,1,1,2,2,2,3,3,3,3,4,4), 
  items=c("A","C","D","B","C","E","A","B","C","E","B","E")
)
(MyList<-split(MyT[,"items"],MyT[,"TID"]))
(MyTrans<-as(MyList,"transactions"))

###########生成transactoins对象：方式四
MyTrans<-read.transactions(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\事务原始数据.txt",format="basket",sep=",")
MyTrans<-read.transactions(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\事务表数据.txt",format="single",cols=c("TID","ITEMS"),sep="	",header = TRUE)


##################apriori算法：搜索频繁项集
MyTrans<-read.transactions(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\事务原始数据.txt",format="basket",sep=",")
#apriori(data=transactions类对象名,parameter=NULL,appearance=NULL)
#parameter是参数列表放在list中。
#support指定最小支持度阈值默认0.1,confidence指定最小指定度阈值,minlen指定关联规则所包含的最小项目数最小值为1
#max默认值10.
#target,指定最终会给出怎样的搜索结果"rules"表示给出简单关联规则,"frequent itemsets"给出所有频繁项集
#"maximally frequent itemsets"表示给出最大频繁项集
#appearance关于关联约束的列表。lhs仅给出规则前项中符合指定特征的规则。
#rhs指定仅给出规则后项中符合指定特征的规则,items针对频繁项集指定仅给出某频繁项集
#none,指定仅给出不包含某些特征的项集或规则
#default指定关联约束列表中没有明确指定的特征关联。

MyRules<-apriori(data=MyTrans,parameter=list(support=0.5,confidence=0.6,target="frequent itemsets"))
inspect(MyRules)#浏览频繁集项
MyRules<-apriori(data=MyTrans,parameter=list(support=0.5,confidence=0.6,target="maximally frequent itemsets"))
inspect(MyRules)

##################apriori算法：生成关联规则
MyTrans<-read.transactions(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\事务原始数据.txt",format="basket",sep=",")
MyRules<-apriori(data=MyTrans,parameter=list(support=0.5,confidence=0.6,target="rules"))
inspect(MyRules)
#lhs          rhs    support confidence lift
#规则前项  规则后项  置信度   置信度    提升度
size(x=MyRules)
MyRules.sorted<-sort(x=MyRules,by="lift",decreasing=TRUE)
inspect(MyRules.sorted)

##################apriori算法：筛选关联规则
#subset筛选规则
#包含两个项目
MyRules.D<-subset(x=MyRules,subset=size(MyRules)==2)
inspect(MyRules.D)
#提升度大于1
MyRules.D<-subset(x=MyRules,subset=slot(object=MyRules,name="quality")$lift>1)
inspect(MyRules.D)
#MyRules.D<-subset(x=MyRules,subset=quality(MyRules)$lift>1)  
#a<-as(MyRules,"data.frame") 
#a[a$lift>1,]

MyRules<-apriori(data=MyTrans,parameter=list(support=0.5,confidence=0.6,target="rules"),
                 appearance=list(lhs=c("B"),default="rhs"))
inspect(MyRules)


#############可视化频繁项集
library("arulesViz")
MyTrans<-read.transactions(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\事务原始数据.txt",format="basket",sep=",")
MyRules<-apriori(data=MyTrans,parameter=list(support=0.5,confidence=0.6,target="frequent itemsets"))
inspect(MyRules)
plot(x=MyRules,method="graph",control=list(main="示例的频繁项集可视化结果"))

#############可视化关联规则
MyTrans<-read.transactions(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\事务原始数据.txt",format="basket",sep=",")
MyRules<-apriori(data=MyTrans,parameter=list(support=0.5,confidence=0.6,target="rules"))
plot(MyRules,method="grouped")
#线粗代表支持度大小,灰度深浅表示提升度
plot(MyRules,method="paracoord")

plot(MyRules,method="graph",control=list(arrowSize=2,main="示例的关联规则可视化结果"))
