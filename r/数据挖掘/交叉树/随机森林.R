#########################随机森林
library("randomForest")
MailShot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\邮件营销数据.txt",header=TRUE)
MailShot<-MailShot[,-1]
set.seed(12345)
#proximity临近
(rFM<-randomForest(MAILSHOT~.,data=MailShot,importance=TRUE,proximity=TRUE))
#各观测的各类别预测概率
head(rFM$votes)     
#各观测作为袋外观测的次数
head(rFM$oob.times)       
DrawL<-par()
par(mfrow=c(2,1),mar=c(5,5,3,1))

#等价于对err.rate画图
plot(rFM,main="随机森林的OOB错判率和决策树棵树")
#探测边界点
plot(margin(rFM),type="h",main="边界点探测",xlab="观测序列",ylab="比率差")
par(DrawL)
#随机森林对全部观测做预测
Fit<-predict(rFM,MailShot)
#混淆矩阵
ConfM5<-table(MailShot$MAILSHOT,Fit)

#错判率
(E5<-(sum(ConfM5)-sum(diag(ConfM5)))/sum(ConfM5))

#浏览各个树的叶节点个数
head(treesize(rFM)) 
#提取第1棵树的部分信息
head(getTree(rfobj=rFM,k=1,labelVar=TRUE))
barplot(rFM$importance[,3],main="输入变量重要性测度(预测精度变化)指标柱形图")
box()
importance(rFM,type=1)
varImpPlot(x=rFM, sort=TRUE, n.var=nrow(rFM$importance),main="输入变量重要性测度散点图") 
