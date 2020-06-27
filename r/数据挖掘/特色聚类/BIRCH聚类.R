###############模拟数据的BIRCH聚类
#无法实现新版本的R已经不支持。
#以后准备使用python实现
library("birch")
library(MASS)
set.seed(12345) 
Data<-mvrnorm(1000,mu=rep(0,2),Sigma=diag(1,2))
Data<-rbind(Data,mvrnorm(1000,mu=rep(10,2),Sigma=diag(0.1,2)+0.9))
par(mfrow=c(2,2))
plot(Data,main="样本观测点的分布",xlab="x1",ylab="x2")  
Mybirch<-birch(x=Data,radius=5,keeptree=TRUE)   
(OutBirch<-birch.getTree(Mybirch))           
plot(OutBirch,main="BIRCH聚类解",xlab="x1",ylab="x2")           

set.seed(12345) 
NewData<-mvrnorm(10,mu=rep(7,2),Sigma=diag(0.1,2)+0.9)  
plot(Data,main="样本观测点的分布",xlab="x1",ylab="x2") 
points(NewData,col=2)
birch.addToTree(x=NewData,birchObject=OutBirch)   
OutBirch<-birch.getTree(birchObject=OutBirch)   
plot(OutBirch,main="BIRCH聚类解",xlab="x1",ylab="x2")

set.seed(12345)
kOut<-kmeans.birch(OutBirch,center=4,nstart=2)   
plot(OutBirch,col=kOut$clust$sub,main="BIRCH聚类解优化",xlab="x1",ylab="x2")      
plot(Data,col=kOut$clust$obs,main="最终聚类解",xlab="x1",ylab="x2")    
bDist<-dist.birch(OutBirch)     
hc<-hclust(bDist,method="complete")    
plot(hc,main="BIRCH聚类解的距离树形图")
box()
hc<-cutree(hc,k=4)      
plot(kOut$clust$sub,pch=hc,main="K-Means和分层聚类的优化结果对比",ylab="K-Means聚类")   #对比两种优化结果
birch.killTree(birchObject=OutBirch)

################BIRCH聚类应用
TrainData<-read.table(file="员工培训数据.txt",header=TRUE)
CluData<-as.matrix(TrainData[,1:5])
par(mfrow=c(2,2))
plot(CluData)
set.seed(12345)
Mybirch<-birch(x=CluData,radius=0.3,keeptree=FALSE)
plot(Mybirch)
set.seed(12345)
kOut<-kmeans.birch(Mybirch,center=4)
plot(Mybirch,col=kOut$clust$sub)
TrainData$memb<-kOut$clust$obs
plot(jitter(TrainData$memb),TrainData$X6,col=TrainData$memb,xlab="类成员",ylab="受教育程度")