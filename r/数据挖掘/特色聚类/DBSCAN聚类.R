################ DBSCAN聚类示例
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\模式识别数据.txt",sep=",",head=TRUE)
library("fpc")
par(mfrow=c(2,3))
plot(Data,cex=0.5,main="观测点的分布图")

#dbscan(data=矩阵或数据框,eps=n,MinPts=5,scale = FALSE/TRUE)
#eps邻域半径MinPts邻域包含最小点个数
#因范围小无法找到更多噪点,所以所有观测都是噪点
(DBS1<-dbscan(data=Data,eps=0.2,MinPts=200,scale = FALSE)) 
plot(DBS1,Data,cex=0.5,main="DBSCAN聚类(eps=0.2,MinPts=200)")

#因较大范围内找到邻域点,所以只有1个噪点
(DBS2<-dbscan(data=Data,eps=0.5,MinPts=80,scale = FALSE)) 
plot(DBS2,Data,cex=0.5,main="DBSCAN聚类(eps=0.5,MinPts=80)")


(DBS3<-dbscan(data=Data,eps=0.2,MinPts=100,scale = FALSE))
plot(DBS3,Data,cex=0.5,main="DBSCAN聚类(eps=0.2,MinPts=100)")

(DBS4<-dbscan(data=Data,eps=0.5,MinPts=300,scale = FALSE))
plot(DBS4,Data,cex=0.5,main="DBSCAN聚类(eps=0.5,MinPts=300)")

#有62个噪点分为4类。
(DBS5<-dbscan(data=Data,eps=0.2,MinPts=30,scale = FALSE))
plot(DBS5,Data,cex=0.5,main="DBSCAN聚类(eps=0.2,MinPts=30)")