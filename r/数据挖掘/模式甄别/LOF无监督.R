#############模拟数据异常点甄别：LOF法(无监督)
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\模式甄别模拟数据1.txt",header=TRUE,sep=",")
library("DMwR")

lof.scores<-lofactor(data=Data[,-3],k=20)
par(mfrow=c(2,2)) 
Data$lof.scores<-lof.scores
Data.Sort<-Data[order(x=Data$lof.scores,decreasing=TRUE),]
P<-0.1
N<-length(Data[,1])
NoiseP<-head(Data.Sort,trunc(N*P))
colP<-ifelse(1:N %in% rownames(NoiseP),2,1)
#决策精度大于80%
plot(Data[,1:2],main="LOF的模式诊断结果",xlab="",ylab="",pch=Data[,3]+1,cex=0.8,col=colP)
library("ROCR")
pd<-prediction(Data$lof.scores,Data$y)
pf1<-performance(pd,measure="rec",x.measure="rpp") #y轴为回溯精度，X轴为预测的模式占总样本的比例
pf2<-performance(pd,measure="prec",x.measure="rec")   #y轴为决策精度，X轴为回溯精度
plot(pf1,main="模式甄别的累计回溯精度曲线")
plot(pf2,main="模式甄别的决策精度和回溯精度曲线")
