#################模拟数据异常点甄别：朴素贝叶斯(有监督)
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\模式甄别模拟数据2.txt",header=TRUE,sep=",")
library("klaR")
BayesModel<-NaiveBayes(x=Data[,1:2],grouping=factor(Data[,3]))  #输出变量应为因子
BayesModel$apriori   #显示先验概率
BayesModel$tables    #显示各分布的参数估计值
plot(BayesModel)    #可视化各个分布
BayesFit<-predict(object=BayesModel,newdata=Data[,1:2])    #预测
head(BayesFit$class)  #显示预测类别
head(BayesFit$posterior)     #显示后验概率
par(mfrow=c(2,2))
plot(Data[,1:2],main="朴素贝叶斯分类的模式甄别结果",xlab="x1",ylab="x2",
     pch=Data[,3]+1,col=as.integer(as.vector(BayesFit$class))+1,cex=0.8)  #可视化观测点分布特征
library("ROCR")
pd<-prediction(BayesFit$posterior[,2],Data$y)
pf1<-performance(pd,measure="rec",x.measure="rpp") #y轴为回溯精度，X轴为预测的模式占总样本的比例
pf2<-performance(pd,measure="prec",x.measure="rec")   #y轴为决策精度，X轴为回溯精度
#若将风险排名20%的观测预测为模式,回溯精度可达100%。决策精度大于60%
plot(pf1,main="模式甄别的累计回溯精度曲线")
plot(pf2,main="模式甄别的决策精度和回溯精度曲线")
