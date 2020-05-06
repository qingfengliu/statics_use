############天猫数据加工
#按书中说法添加了若干派生变量
#消费者活跃度BuyDNactDN
#活跃度ActDNTotalDN
#成交有效度BuyBBrand
#活动有效度BuyHit
#定义标签变量是否有订单成交BuyOrNot
GetRatio<-function(data,days){    
  data<-within(data,{
    BuyHit<-ifelse(data$hitN!=0,round(data$buyN/data$hitN*100,2),NA)  
    BuyBBrand<-ifelse(data$brandN!=0,round(data$buyBrandN/data$brandN*100,2),NA)   
    ActDNTotalDN<-round(data$actDateN/days*100,2)            
    BuyDNactDN<-ifelse(data$actDateN!=0,round(data$buyDateN/data$actDateN*100,2),NA)  
    BuyOrNot<-sapply(data$buyN,FUN=function(x) ifelse(x!=0,"1","0")) 
    BuyOrNot<-as.factor(BuyOrNot)
  })
  return(data)
}
Tmall_train<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\天猫_Train.txt",header=TRUE,sep=",")
Tmall_train<-GetRatio(Tmall_train,92)
Tmall_train<-Tmall_train[complete.cases(Tmall_train),]     #只取完整观测数据
Tmall_test<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\天猫_Test.txt",header=TRUE,sep=",")
Tmall_test<-GetRatio(Tmall_test,31)
Tmall_test<-Tmall_test[complete.cases(Tmall_test),]   

#保存中间数据,这步书中没有
write.table(Tmall_train[,-(1:9)],file="天猫_Train_1.txt",sep=",",quote=FALSE,append=FALSE,row.names=FALSE,col.names=TRUE)
write.table(Tmall_test[,-(1:9)],file="天猫_Test_1.txt",sep=",",quote=FALSE,append=FALSE,row.names=FALSE,col.names=TRUE)

####天猫数据KNN分类讨论变量重要性
library("class")  
par(mfrow=c(2,2))
errRatio<-vector()
for(i in 1:30){   
  #Tmall_train[,c(-(1:9))]，这个语法是去除元数据第1-9列
  fit<-knn(train=Tmall_train[,c(-(1:9))],test=Tmall_test[,c(-(1:9))],cl=Tmall_train[,10],k=i)
  CT<-table(Tmall_test[,10],fit)
  errRatio<-c(errRatio,(1-sum(diag(CT))/sum(CT))*100)   
}

plot(errRatio,type="l",xlab="参数K",ylab="错判率(%)",main="参数K与错判率",cex.main=0.8)

errDelteX<-errRatio[7]   #去合适k=7

#在选取k=7的情况下依次剔除第11-14列。然后看错判率变化情况。可看出剔除第12列
#ActDNTotalDN活跃度后错判率明显下降。
for(i in -11:-14){
  fit<-knn(train=Tmall_train[,c(-(1:9),i)],test=Tmall_test[,c(-(1:9),i)],cl=Tmall_train[,10],k=7)
  CT<-table(Tmall_test[,10],fit)
  errDelteX<-c(errDelteX,(1-sum(diag(CT))/sum(CT))*100)
}
plot(errDelteX,type="l",xlab="剔除变量",ylab="剔除错判率(%)",main="剔除变量与错判率(K=7)",cex.main=0.8)
xTitle=c("1:全体变量","2:消费活跃度","3:活跃度","4:成交有效度","5:活动有效度")
legend("topright",legend=xTitle,title="变量说明",lty=1,cex=0.6) 
#这里依据FI定义计算权重
FI<-errDelteX[-1]+1/4   
wi<-FI/sum(FI)
#将计算出的权重绘制成饼图
GLabs<-paste(c("消费活跃度","活跃度","成交有效度","活动有效度"),round(wi,2),sep=":")
pie(wi,labels=GLabs,clockwise=TRUE,main="输入变量权重",cex.main=0.8)
#算出权重并没用上？




ColPch=as.integer(as.vector(Tmall_test[,10]))+1
plot(Tmall_test[,c(11,13)],pch=ColPch,cex=0.7,xlim=c(0,50),ylim=c(0,50),col=ColPch,
     xlab="消费活跃度",ylab="成交有效度",main="二维特征空间中的观测",cex.main=0.8)
