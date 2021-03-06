#####################K-Means聚类应用
PoData<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\环境污染数据.txt",header=TRUE)
CluData<-PoData[,2:7]
#############K-Means聚类
set.seed(12345)
CluR<-kmeans(x=CluData,centers=4,iter.max=10,nstart=30)
CluR$size
CluR$centers
###########K-Means聚类结果的可视化 
par(mfrow=c(2,1))
PoData$CluR<-CluR$cluster
plot(PoData$CluR,pch=PoData$CluR,ylab="类别编号",xlab="省市",main="聚类的类成员",axes=FALSE)
par(las=2)
axis(1,at=1:31,labels=PoData$province,cex.axis=0.6)
axis(2,at=1:4,labels=1:4,cex.axis=0.6)
box()
legend("topright",c("第一类","第二类","第三类","第四类"),pch=1:4,cex=0.6)
###########K-Means聚类特征的可视化
plot(CluR$centers[1,],type="l",ylim=c(0,82),xlab="聚类变量",ylab="组均值(类质心)",main="各类聚类变量均值的变化折线图",axes=FALSE)
axis(1,at=1:6,labels=c("生活污水排放量","生活二氧化硫排放量","生活烟尘排放量","工业固体废物排放量","工业废气排放总量","工业废水排放量"),cex.axis=0.6)
box()
lines(1:6,CluR$centers[2,],lty=2,col=2)
lines(1:6,CluR$centers[3,],lty=3,col=3)
lines(1:6,CluR$centers[4,],lty=4,col=4)
legend("topleft",c("第一类","第二类","第三类","第四类"),lty=1:4,col=1:4,cex=0.6)
#解释变量离差平方和占总平方和的64.92%聚类效果不理想。

###########K-Means聚类效果的可视化评价
CluR$betweenss/CluR$totss*100
par(mfrow=c(2,3))
plot(PoData[,c(2,3)],col=PoData$CluR,main="生活污染情况",xlab="生活污水排放量",ylab="生活二氧化硫排放量")
points(CluR$centers[,c(1,2)],col=rownames(CluR$centers),pch=8,cex=2)
plot(PoData[,c(2,4)],col=PoData$CluR,main="生活污染情况",xlab="生活污水排放量",ylab="生活烟尘排放量")
points(CluR$centers[,c(1,3)],col=rownames(CluR$centers),pch=8,cex=2)
plot(PoData[,c(3,4)],col=PoData$CluR,main="生活污染情况",xlab="生活二氧化硫排放量",ylab="生活烟尘排放量")
points(CluR$centers[,c(2,3)],col=rownames(CluR$centers),pch=8,cex=2)

plot(PoData[,c(5,6)],col=PoData$CluR,main="工业污染情况",xlab="工业固体废物排放量",ylab="工业废气排放总量")
points(CluR$centers[,c(4,5)],col=rownames(CluR$centers),pch=8,cex=2)
plot(PoData[,c(5,7)],col=PoData$CluR,main="工业污染情况",xlab="工业固体废物排放量",ylab="工业废水排放量")
points(CluR$centers[,c(4,6)],col=rownames(CluR$centers),pch=8,cex=2)
plot(PoData[,c(6,7)],col=PoData$CluR,main="工业污染情况",xlab="工业废气排放总量",ylab="工业废水排放量")
points(CluR$centers[,c(5,6)],col=rownames(CluR$centers),pch=8,cex=2)
#从各个截面上看质心相距较远,个观测点重合不严重聚类可接受