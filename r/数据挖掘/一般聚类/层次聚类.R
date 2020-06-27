################层次聚类
PoData<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\环境污染数据.txt",header=TRUE)
CluData<-PoData[,2:7]
#计算欧式距离
DisMatrix<-dist(CluData,method="euclidean")
#聚类函数
#hclust(d=矩阵距离,method=聚类方法)
#single表示最近邻法。
#complete表示组内离差平方和法
#average表示组间平均锁链法
#centroid质心法
#ward表示离差平方和法
#返回一参数列表，其中height的成分记录了聚类过程中聚成n-1,n-2,...,1类时的最小类间距离。
#层次聚类过程决定了这个距离是在不断增大的
CluR<-hclust(d=DisMatrix,method="ward.D")

###############层次聚类的树形图
plot(CluR,labels=PoData[,1])
box()#边框
###########层次聚类的碎石图
plot(CluR$height,30:1,type="b",cex=0.7,xlab="距离测度",ylab="聚类数目")

######取4类的聚类解并可视化
par(mfrow=c(2,1))
#cutree(层次聚类结果,聚类数目)
PoData$memb<-cutree(CluR,k=4)   #分成4类
table(PoData$memb)
plot(PoData$memb,pch=PoData$memb,ylab="类别编号",xlab="省市",main="聚类的类成员",axes=FALSE)

par(las=2)
axis(1,at=1:31,labels=PoData$province,cex.axis=0.6)
axis(2,at=1:4,labels=1:4,cex.axis=0.6)
box()
