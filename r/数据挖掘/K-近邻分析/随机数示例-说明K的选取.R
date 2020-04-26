
##########KNN分类
set.seed(12345)   #随机数种子
x1<-runif(60,-1,1)  # 
x2<-runif(60,-1,1)  # 
y<-sample(c(0,1),size=60,replace=TRUE,prob=c(0.3,0.7))  #从0,1两个数中做放回抽样，最后结果0,1占比为3:7    
Data<-data.frame(Fx1=x1,Fx2=x2,Fy=y)  #将x1,x2,y放到一个dataframe里。将y抽象成因变量。x,y自变量
SampleId<-sample(x=1:60,size=18)  #相当于生成了一个随机数列，数量为18范围为1:60
DataTest<-Data[SampleId,]   
DataTrain<-Data[-SampleId,]  #这里划分了训练集和测试集。这个方法很巧妙不知道python能否达到

par(mfrow=c(2,2),mar=c(4,6,4,4))  #生成绘图空间
plot(Data[,1:2],pch=Data[,3]+1,cex=0.8,xlab="x1",ylab="x2",main="全部样本") #散点图
plot(DataTrain[,1:2],pch=DataTrain[,3]+1,cex=0.8,xlab="x1",ylab="x2",main="训练样本和测试样本")
points(DataTest[,1:2],pch=DataTest[,3]+16,col=2,cex=0.8) #与上边的语句合成了,将测试样本和训练样本区分出来

library("class")
errRatio<-vector()  
for(i in 1:30){    #计算全部观测的错判率向量， k取1-30
  KnnFit<-knn(train=Data[,1:2],test=Data[,1:2],cl=as.factor(Data[,3]),k=i)
  CT<-table(Data[,3],KnnFit)  #计算混淆矩阵
  errRatio<-c(errRatio,(1-sum(diag(CT))/sum(CT))*100)   #计算错判率     
}
#上边主要取k 1-30来看误差 然后看下哪个误差较低
plot(errRatio,type="l",xlab="近邻个数K",ylab="错判率(%)",main="近邻数K与错判率",ylim=c(0,80))
#上边的k的不同值的误差放到图里方便看。从结果上看,错判率是很大的可能由于3:7的比例？

errRatio1<-vector()   
for(i in 1:30){  #使用旁置法选择k
  KnnFit<-knn(train=DataTrain[,1:2],test=DataTest[,1:2],cl=as.factor(DataTrain[,3]),k=i) 
  CT<-table(DataTest[,3],KnnFit)    
  errRatio1<-c(errRatio1,(1-sum(diag(CT))/sum(CT))*100)    
}
lines(1:30,errRatio1,lty=2,col=2)

set.seed(12345)
errRatio2<-vector()   
for(i in 1:30){   #留一法
  KnnFit<-knn.cv(train=Data[,1:2],cl=as.factor(Data[,3]),k=i) 
  CT<-table(Data[,3],KnnFit)  
  errRatio2<-c(errRatio2,(1-sum(diag(CT))/sum(CT))*100)     
}
lines(1:30,errRatio2,col=2)
#以上对各种方法的误判率，适用于分类变量


##############以下，验证的是回归预测算的是均方误差
set.seed(12345)
x1<-runif(60,-1,1)  
x2<-runif(60,-1,1)  
y<-runif(60,10,20)   
Data<-data.frame(Fx1=x1,Fx2=x2,Fy=y)
SampleId<-sample(x=1:60,size=18)  
DataTest<-Data[SampleId,]  
DataTrain<-Data[-SampleId,]  
mseVector<-vector()    
for(i in 1:30){
  KnnFit<-knn(train=DataTrain[,1:2],test=DataTest[,1:2],cl=DataTrain[,3],k=i,prob=FALSE) 
  KnnFit<-as.double(as.vector(KnnFit))   
  mse<-sum((DataTest[,3]-KnnFit)^2)/length(DataTest[,3])   
  mseVector<-c(mseVector,mse)
}
plot(mseVector,type="l",xlab="近邻个数K",ylab="均方误差",main="近邻数K与均方误差",ylim=c(0,80))