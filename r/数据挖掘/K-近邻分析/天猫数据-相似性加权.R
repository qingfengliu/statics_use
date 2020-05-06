############天猫数据加权KNN分类
library("kknn")
par(mfrow=c(2,1))
Tmall_train<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\天猫_Train_1.txt",header=TRUE,sep=",")
Tmall_train$BuyOrNot<-as.factor(Tmall_train$BuyOrNot)
fit<-train.kknn(formula=BuyOrNot~.,data=Tmall_train,kmax=11,distance=2,kernel=c("rectangular","triangular","gaussian"),na.action=na.omit())
plot(fit$MISCLASS[,1]*100,type="l",main="不同核函数和近邻个数K下的错判率曲线图",cex.main=0.8,xlab="近邻个数K",ylab="错判率(%)")
lines(fit$MISCLASS[,2]*100,lty=2,col=1)
lines(fit$MISCLASS[,3]*100,lty=3,col=2)
legend("topleft",legend=c("rectangular","triangular","gaussian"),lty=c(1,2,3),col=c(1,1,2),cex=0.7)   #给出图例

Tmall_test<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\天猫_Test_1.txt",header=TRUE,sep=",")
Tmall_test$BuyOrNot<-as.factor(Tmall_test$BuyOrNot)
fit<-kknn(formula=BuyOrNot~.,train=Tmall_train,test=Tmall_test,k=7,distance=2,kernel="gaussian",na.action=na.omit())
CT<-table(Tmall_test[,1],fit$fitted.values)
errRatio<-(1-sum(diag(CT))/sum(CT))*100

library("class")
fit<-knn(train=Tmall_train,test=Tmall_test,cl=Tmall_train$BuyOrNot,k=7)
CT<-table(Tmall_test[,1],fit)
errRatio<-c(errRatio,(1-sum(diag(CT))/sum(CT))*100)
errGraph<-barplot(errRatio,main="加权K近邻法与K近邻法的错判率对比图(K=7)",cex.main=0.8,xlab="分类方法",ylab="错判率(%)",axes=FALSE)
axis(side=1,at=c(0,errGraph,3),labels=c("","加权K-近邻法","K-近邻法",""),tcl=0.25)
axis(side=2,tcl=0.25)