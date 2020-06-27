################模拟数据异常点甄别：DB法(无监督)
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\模式甄别模拟数据1.txt",header=TRUE,sep=",")
N<-length(Data[,1])
DistM<-as.matrix(dist(Data[,1:2]))
par(mfrow=c(2,2)) 
(D<-quantile(x=DistM[upper.tri(DistM,diag=FALSE)],prob=0.75))  #计算距离的分位数作为阈值D
for(i in 1:N){
  x<-as.vector(DistM[i,])
  Data$DB.scores[i]<-length(which(x>D))/N    #计算观测x与其他观测间的距离大于阈值D的个数占比
}
Data.Sort<-Data[order(x=Data$DB.score,decreasing=TRUE),]
#前10%作为可能
P<-0.1
NoiseP<-head(Data.Sort,trunc(N*P))
colP<-ifelse(1:N %in% rownames(NoiseP),2,1)
plot(Data[,1:2],main=paste("DB的模式诊断结果:p=",P,sep=""),xlab="x1",ylab="x2",pch=Data[,3]+1,cex=0.8,col=colP)

library("ROCR")
pd<-prediction(Data$DB.scores,Data$y)
pf1<-performance(pd,measure="rec",x.measure="rpp") #y轴为回溯精度，X轴为预测的模式占总样本的比例
pf2<-performance(pd,measure="prec",x.measure="rec")   #y轴为决策精度，X轴为回溯精度
plot(pf1,main="模式甄别的累计回溯精度曲线")
plot(pf2,main="模式甄别的决策精度和回溯精度曲线")
P<-0.25
NoiseP<-head(Data.Sort,trunc(N*P))
colP<-ifelse(1:N %in% rownames(NoiseP),2,1)
plot(Data[,1:2],main=paste("DB的模式诊断结果:p=",P,sep=""),xlab="x1",ylab="x2",pch=Data[,3]+1,cex=0.8,col=colP)
