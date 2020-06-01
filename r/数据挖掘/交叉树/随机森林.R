#########################随机森林
library("randomForest")
MailShot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\邮件营销数据.txt",header=TRUE)
MailShot<-MailShot[,-1]
set.seed(12345)
#proximity临近
#randomForest(输出变量~输入变量,data=数据框名,mtry=k,ntree=M,importance=TRUE)
#参数mtry用于指定变量子集Θi包含的输入变量的个数。若输出变量为数值型变量,基础学习器为回归树,
#k默认为√p。若输出变量为数值型变量,基础学习器为回归树,k默认为P/3。
#参数ntree用于指定随机森林包含M棵决策树,默认为500。
#参数importance=TRUE表示计算输入变量对输出变量的重要性的测度值。
#randomForest函数的返回值为列表:
#predicted各决策树对其袋外观测值的预测类别的众数,或预测值的平均。
#confusion基于袋外观测的混淆矩阵。
#votes适用于分类树。给出各预测类别的概率值,即随机森林中有多大比例的决策树投票给第i类别。
#oob.times各个观测作为袋外观测的次数,即在重抽样自举中有多少次未进入自举样本。
#它会影响基于袋外观测的误差结果。
#err.rate:随机森林对袋外观测的整体预测错误率,以及对各个类别的预测错误率。
#importance:输入变量重要性的测度矩阵。具体说明可以去书中对照。
(rFM<-randomForest(MAILSHOT~.,data=MailShot,importance=TRUE,proximity=TRUE))
#本例的袋外观测错误率高达42%,NO的错误率达33%,YES错误率高达54%
#模型不理想

#各观测的各类别预测概率
head(rFM$votes)     
#各观测作为袋外观测的次数
head(rFM$oob.times)       
DrawL<-par()
par(mfrow=c(2,1),mar=c(5,5,3,1))

#等价于对err.rate画图
plot(rFM,main="随机森林的OOB错判率和决策树棵树")
#图中黑线代表整体错误率,红线为对NO的判错率,绿线为对YES的判错率。
#可见当决策树达到380后,各判错率基保持稳定。本例的ntree可设置为380

#探测边界点
plot(margin(rFM),type="h",main="边界点探测",xlab="观测序列",ylab="比率差")
#边界附近点的判断依据是利用,对于某个观测点投票给正确类别的树的比率减去投票给其他
#类别的树的比率。也就是说比率差接近正数,代表判断正确,负数为判断错误。
#越接近0则越在边界,不易判断类别。

par(DrawL)


#随机森林对全部观测做预测
Fit<-predict(rFM,MailShot)
#混淆矩阵
ConfM5<-table(MailShot$MAILSHOT,Fit)

#错判率
(E5<-(sum(ConfM5)-sum(diag(ConfM5)))/sum(ConfM5))
#整体的判错率是较低的为0.03


#浏览各个树的叶节点个数
#treesize(随机森林结果对象名,terminal=TRUE/FALSE)
#terminal=TRUE代表仅统计决策树的叶节点个数。
#false代表统计所有节点的个数
head(treesize(rFM)) 

#提取第1棵树的部分信息
#labelVar=FALSE/TRUE，TRUE代表所有输入变量中找到当前节点的最佳分组。
head(getTree(rfobj=rFM,k=1,labelVar=TRUE))
#返回值书中也有比较详细的介绍由于比较复杂不在这里提到。

#变量重要性的测度。
#柱图
barplot(rFM$importance[,3],main="输入变量重要性测度(预测精度变化)指标柱形图")
box()

#点图
importance(rFM,type=1)
varImpPlot(x=rFM, sort=TRUE, n.var=nrow(rFM$importance),main="输入变量重要性测度散点图") 
