####################可视化模拟数据1
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\模式甄别模拟数据1.txt",header=TRUE,sep=",")
head(Data)
plot(Data[,1:2],main="样本观测点的分布",xlab="x1",ylab="x2",pch=Data[,3]+1,cex=0.8)  #可视化观测点分布特征
####################可视化模拟数据2
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\模式甄别模拟数据2.txt",header=TRUE,sep=",")
head(Data)
plot(Data[,1:2],main="样本观测点的分布",xlab="x1",ylab="x2",pch=Data[,3]+1,cex=0.8)  #可视化观测点分布特征

############模拟数据异常点甄别：EM聚类(无监督)
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\模式甄别模拟数据1.txt",header=TRUE,sep=",")
library("mclust") 

EMfit<-Mclust(data=Data[,-3])  
par(mfrow=c(2,2))
Data$ker.scores<-EMfit$uncertainty  
Data.Sort<-Data[order(x=Data$ker.scores,decreasing=TRUE),]
P<-0.1                     
N<-length(Data[,1])         
NoiseP<-head(Data.Sort,trunc(N*P))
colP<-ifelse(1:N %in% rownames(NoiseP),2,1)
plot(Data[,1:2],main="EM聚类的模式诊断结果(10%)",xlab="x1",ylab="x2",pch=Data[,3]+1,cex=0.8,col=colP)

library("ROCR")
pd<-prediction(Data$ker.scores,Data$y)
pf1<-performance(pd,measure="rec",x.measure="rpp") 
pf2<-performance(pd,measure="prec",x.measure="rec")   
plot(pf1,main="模式甄别的累计回溯精度曲线")
plot(pf2,main="模式甄别的决策精度和回溯精度曲线")
P<-0.25
NoiseP<-head(Data.Sort,trunc(N*P))
colP<-ifelse(1:N %in% rownames(NoiseP),2,1)
plot(Data[,1:2],main="EM聚类的模式诊断结果(25%)",xlab="x1",ylab="x2",pch=Data[,3]+1,cex=0.8,col=colP)

