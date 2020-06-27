#################模拟数据异常点甄别：Logistic回归(有监督)
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\模式甄别模拟数据2.txt",header=TRUE,sep=",")
#glm logistic函数
#glm(R公式,data=数据框名,family=binomial(link="logit"))
(LogModel<-glm(factor(y)~.,data=Data,family=binomial(link="logit")))  
#predict logic回归预测
LogFit<-predict(object=LogModel,newdata=Data,type="response")
Data$Log.scores<-LogFit
library("ROCR")
par(mfrow=c(2,2))
pd<-prediction(Data$Log.scores,Data$y)
pf1<-performance(pd,measure="rec",x.measure="rpp") #y轴为回溯精度，X轴为预测的模式占总样本的比例
pf2<-performance(pd,measure="prec",x.measure="rec")   #y轴为决策精度，X轴为回溯精度
plot(pf1,main="模式甄别的累计回溯精度曲线",print.cutoffs.at=c(0.15,0.1))
plot(pf2,main="模式甄别的决策精度和回溯精度曲线")
Data.Sort<-Data[order(x=Data$Log.scores,decreasing=TRUE),]
P<-0.20
N<-length(Data[,1])
NoiseP<-head(Data.Sort,trunc(N*P))
colP<-ifelse(1:N %in% rownames(NoiseP),2,1)
plot(Data[,1:2],main="Logistic回归的模式甄别结果(20%)",xlab="x1",ylab="x2",pch=Data[,3]+1,cex=0.8,col=colP)
P<-0.30
NoiseP<-head(Data.Sort,trunc(N*P))
colP<-ifelse(1:N %in% rownames(NoiseP),2,1)
plot(Data[,1:2],main="Logistic回归的模式甄别结果(30%)",xlab="x1",ylab="x2",pch=Data[,3]+1,cex=0.8,col=colP)

#决策精度较低,未做模型合理检验，并且非平衡数据