###################实例数据的EM聚类
PoData<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\环境污染数据.txt",header=TRUE)
CluData<-PoData[,2:7]
library("mclust") 
EMfit<-Mclust(data=CluData)  
summary(EMfit)
#BIC图展示不同聚类方法的K值与BIC值的折线图,取BIC值值最大的。这里与书中的结果不太一样。
#取EEV的结果
plot(EMfit,"BIC")
plot(EMfit,"classification")

