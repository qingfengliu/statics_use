################天猫数据SVM
Tmall_train<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\天猫_Train_1.txt",header=TRUE,sep=",")
Tmall_train$BuyOrNot<-as.factor(Tmall_train$BuyOrNot)
set.seed(12345)
library("e1071")
tObj<-tune.svm(BuyOrNot~.,data=Tmall_train,type="C-classification",kernel="radial",gamma=10^(-6:-3),cost=10^(-3:2))
plot(tObj,xlab=expression(gamma),ylab="损失惩罚参数C",
     main="不同参数组合下的预测错误率",nlevels=10,color.palette=terrain.colors)
#由等高图可看出C取100,gamma取0.001最好
BestSvm<-tObj$best.model
summary(BestSvm)

#错判率为0.084
Tmall_test<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\天猫_Test_1.txt",header=TRUE,sep=",")
Tmall_test$BuyOrNot<-as.factor(Tmall_test$BuyOrNot)
yPred<-predict(BestSvm,Tmall_test)
(ConfM<-table(yPred,Tmall_test$BuyOrNot))
(Err<-(sum(ConfM)-sum(diag(ConfM)))/sum(ConfM))