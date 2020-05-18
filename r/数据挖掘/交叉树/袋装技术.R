#######################建立单个分类树
library("rpart")
library("ipred")
library("rpart.plot")
MailShot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\邮件营销数据.txt",header=TRUE)
MailShot<-MailShot[,-1]#剔除第一列
Ctl<-rpart.control(minsplit=20,maxcompete=4,maxdepth=30,cp=0.01,xval=10)
set.seed(12345)
TreeFit<-rpart(MAILSHOT~.,data=MailShot,method="class",parms=list(split="gini"))
rpart.plot(TreeFit,type=4,branch=0,extra=1)

#利用单棵树对全部观测值进行预测
CFit1<-predict(TreeFit,MailShot,type="class")  #CFit1<-predict(TreeFit,MailShot)
#计算单棵树混淆矩阵
ConfM1<-table(MailShot$MAILSHOT,CFit1)
#计算单棵树错判率
(E1<-(sum(ConfM1)-sum(diag(ConfM1)))/sum(ConfM1))
set.seed(12345)

#利用bagging建立组合分类树
#进行25次重抽样自举,生成25棵树。
#运行结果袋外观测的测试误差为0.447

(BagM1<-bagging(MAILSHOT~.,data=MailShot,nbagg=25,coob=TRUE,control=Ctl))
#利用组合分类树对全部观测值进行预测。
CFit2<-predict(BagM1,MailShot,type="class")
#计算混淆矩阵
ConfM2<-table(MailShot$MAILSHOT,CFit2)
#计算判错判率.明显降低
(E2<-(sum(ConfM2)-sum(diag(ConfM2)))/sum(ConfM2))


#用adabag包里的bagging.
#这个函数会将变量重要性进行归一化处理,减小错判率
detach("package:ipred")
library("adabag")
MailShot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\邮件营销数据.txt",header=TRUE)
MailShot<-MailShot[,-1]
Ctl<-rpart.control(minsplit=20,maxcompete=4,maxdepth=30,cp=0.01,xval=10)
set.seed(12345)
BagM2<-bagging(MAILSHOT~.,data=MailShot,control=Ctl,mfinal = 25)
#这里展示变量的权重
BagM2$importance
CFit3<-predict.bagging(BagM2,MailShot)
#混淆矩阵
CFit3$confusion
#这里给出的错判率好像不一样
CFit3$error


