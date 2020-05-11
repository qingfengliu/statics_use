#######################建立单个分类树
library("rpart")
library("ipred")
library("rpart.plot")
MailShot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\邮件营销数据.txt",header=TRUE)
MailShot<-MailShot[,-1]
Ctl<-rpart.control(minsplit=20,maxcompete=4,maxdepth=30,cp=0.01,xval=10)
set.seed(12345)
TreeFit<-rpart(MAILSHOT~.,data=MailShot,method="class",parms=list(split="gini"))
rpart.plot(TreeFit,type=4,branch=0,extra=1)


CFit1<-predict(TreeFit,MailShot,type="class")  #CFit1<-predict(TreeFit,MailShot)
ConfM1<-table(MailShot$MAILSHOT,CFit1)
(E1<-(sum(ConfM1)-sum(diag(ConfM1)))/sum(ConfM1))
set.seed(12345)

(BagM1<-bagging(MAILSHOT~.,data=MailShot,nbagg=25,coob=TRUE,control=Ctl))
CFit2<-predict(BagM1,MailShot,type="class")
ConfM2<-table(MailShot$MAILSHOT,CFit2)
(E2<-(sum(ConfM2)-sum(diag(ConfM2)))/sum(ConfM2))

detach("package:ipred")
library("adabag")
MailShot<-read.table(file="邮件营销数据.txt",header=TRUE)
MailShot<-MailShot[,-1]
Ctl<-rpart.control(minsplit=20,maxcompete=4,maxdepth=30,cp=0.01,xval=10)
set.seed(12345)
BagM2<-bagging(MAILSHOT~.,data=MailShot,control=Ctl,mfinal = 25)
BagM2$importance
CFit3<-predict.bagging(BagM2,MailShot)
CFit3$confusion
CFit3$error