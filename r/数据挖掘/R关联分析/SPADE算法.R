#############SPADE序列序列关联规则示例
library("arulesSequences")
#read_baskets。事物序列数据包含一个序列标识Sid,和一个事物标识Eid
MyTrans<-read_baskets(con="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\事务序列原始数据.txt",sep=",",info=c("sequenceID","eventID"))
#cspade只涉及频繁搜索
#ruleInduction派生规则
MyFsets<-cspade(data=MyTrans,parameter=list(support=0.5))
inspect(MyFsets)
#生成推理规则，体现了SPACE规则的特殊性
MyRules<-ruleInduction(x=MyFsets,confidence=0.3)  
#rule support confidence lift
#1        <{D}> => <{F}>     0.5          1    1
#2      <{D}> => <{B,F}>     0.5          1    1
#比如先D后F的管理置信度为0.5,提升度为1。
MyRules.DF<-as(MyRules,"data.frame")  
MyRules.DF[MyRules.DF$lift>=1,]   

#############序列关联规则应用：
#书中例子里给数出现错误,sequenceID不能出现0
MyTrans<-read_baskets(con="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\网页浏览数据.txt",sep=",",info=c("sequenceID","eventID"))
summary(MyTrans)
MyFsets<-cspade(data=MyTrans,parameter=list(support=0.1))
inspect(MyFsets)
MyRules<-ruleInduction(x=MyFsets,confidence=0.3)  
MyRules.DF<-as(MyRules,"data.frame")  
MyRules.DF[MyRules.DF$lift>=1,]   
