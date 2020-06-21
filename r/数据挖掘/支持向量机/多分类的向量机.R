##############模拟多类别的SVM
#制造多分类的数据。分为3类输出变量的0,1,2。
set.seed(12345)
x<-matrix(rnorm(n=400,mean=0,sd=1),ncol=2,byrow=TRUE)
x[1:100,]<-x[1:100,]+2
x[101:150,]<-x[101:150,]-2
x<-rbind(x,matrix(rnorm(n=100,mean=0,sd=1),ncol=2,byrow=TRUE))
y<-c(rep(1,150),rep(2,50))
y<-c(y,rep(0,50))
x[y==0,2]<-x[y==0,2]+3
data<-data.frame(Fx1=x[,1],Fx2=x[,2],Fy=as.factor(y))
#由图可见是多分类线性不可分问题
plot(data[,2:1],col=as.integer(as.vector(data[,3]))+1,pch=8,cex=0.7,main="训练样本集散点图")

library("e1071")
set.seed(12345)
#用tune.svm函数找到最佳cost为1时。另外summary(BestSvm)的描述信息中没有出现gamma?
#书中例子为5和1
tObj<-tune.svm(Fy~.,data=data,type="C-classification",kernel="radial",
               cost=c(0.001,0.01,0.1,1,5,10,100,1000),gamma=c(0.5,1,2,3,4),scale=FALSE)
BestSvm<-tObj$best.model
summary(BestSvm)
plot(x=BestSvm,data=data,formula=Fx1~Fx2,svSymbol="#",dataSymbol="*",grid=100)

#使用C=5,gamma=1对预测集进行预测,得到判错率为0.084
SvmFit<-svm(Fy~.,data=data,type="C-classification",kernel="radial",cost=5,gamma=1,scale=FALSE)
head(SvmFit$decision.values)
yPred<-predict(SvmFit,data)
(ConfM<-table(yPred,data$Fy))
(Err<-(sum(ConfM)-sum(diag(ConfM)))/sum(ConfM))


