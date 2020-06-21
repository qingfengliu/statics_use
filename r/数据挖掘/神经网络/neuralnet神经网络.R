library("neuralnet")
BuyOrNot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\消费决策数据.txt",header=TRUE)

##########neurealnet建立神经网络
set.seed(12345)

#neurealnet(输出变量~输入变量,data=数据框名,hidden=1,threshold=0.01,stepmax=100000,rep=迭代周期,
#err.fct=误差函数名,linear.output=TRUE,learningrate=学习率,algorithm=算法名)
#hidden=c(3,2,1),表示有3个隐层,第1~3个隐藏分别包含3,2,1个隐节点。
#参数threshold用于迭代停止条件。当权重的最大调整量小于指定值(默认值为0.01)时迭代终止。
#stepmax用于指定迭代停止条件。当迭代次数达到指定次数(默认为100000次)时迭代终止。

#根据是否购买,年龄、性别、收入建模

(BPnet1<-neuralnet(Purchase~Age+Gender+Income,data=BuyOrNot,hidden=2,err.fct="ce",linear.output=FALSE))
BPnet1$result.matrix   #查看连接权重和其他信息
BPnet1$weights         #连接权重列表
BPnet1$startweights 

plot(BPnet1)

#本例有1个隐层和两个隐节点。因为是二分类问题,损失函数采用交互熵,且输出节点的激活函数
#为Sigmoid函数
#结果表明经过15076次迭代(steps)。迭代阶数时损失函数为270.77(error)。权重最大调整值为0.009(reached.threshold)
#result.matrix逐一给出网络节点所有连接权重。如第一个隐节点的偏差权重为85.16。
#年龄、性别、收入与该节点的权重依次为0.89,-45.29,-29.88。

#观测点的广义权重
head(BPnet1$generalized.weights[[1]])
#默认展示前6个观测点的广义权重。年龄在前6个观测处几乎为0.说明年龄不对消费者决策产生影响。

#利用gwplot函数绘制指定输入变量及其在各观测处的广义权重
#gwplot(neuralnet 函数结果对象名,selected.covariate=输入变量名)
par(mfrow=c(2,2))
gwplot(BPnet1,selected.covariate="Age")
gwplot(BPnet1,selected.covariate="Gender")
gwplot(BPnet1,selected.covariate="Income")

#年龄对晓峰决策没有重要影响,排除年龄影响的条件下,不同性别和收入水平人群的购买概率
#是否明显不同呢？
#不同输入变量水平组合下的预测
#compute(neuralnet 函数结果名,covariate=矩阵名)
newData<-matrix(c(39,1,1,39,1,2,39,1,3,39,2,1,39,2,2,39,2,3),nrow=6,ncol=3,byrow=TRUE)
new.output<-compute(BPnet1,covariate=newData)
new.output$net.result
#年龄样本取值为39岁,依次分析性别为1(男),收入依次为1(高),2(中),3(低),以及性别为2(女)
#收入依次1取1，2，3的组合对购买的影响。
#          [,1]
#[1,] 0.2099738
#[2,] 0.2099739
#[3,] 0.4675811
#[4,] 0.3607890
#[5,] 0.4675812
#[6,] 0.4675812
#结果女性明显高于男性,高收入明显低于低收入。

#确定分割概率。
#对于二分类问题,B-P神经网络输出的节点给出的是预测类别为1个概率。通常概率大于分割值
#τ的预测类别为1,小于τ的预测为0.尽管一般情况下分割值τ取0.5但是也有别的情况
#可以通过ROC曲线来寻找分割点。

library("ROCR")
detach("package:neuralnet")

summary(BPnet1$net.result[[1]])#浏览预测概率值
#预测概率值在0.21到0.987之间上四分位数为0.468

#prediction(predictions=概率向量,labels=类别向量)
#目的是将概率值和类别值组织成performance函数要求的对象格式
pred<-prediction(predictions=as.vector(BPnet1$net.result),labels=BPnet1$response)
par(mfrow=c(2,1))

#performance(对象名,measure=缩写1,x.measure=缩写2)
#measure 指定ROC曲线纵坐标(计算方式),x.measure指定横坐标
#keyiqu"tpr"，"fpr"
#画图使用plot函数plot(对象名,colorize=FALSE/TRUE,print.cutoffs.at=c())
#colorize在右侧以不同颜色表示分割τ的大小默认取false
#print.cutoffs.at在ROC曲线上标出各分割点位置。
perf<-performance(pred,measure="tpr",x.measure="fpr")
plot(perf,colorize=TRUE,print.cutoffs.at=c(0.2,0.45,0.46,0.47))

#acc表示预测精度,等于(TP+TN)/N。下边为了看出精度的变化。
#
perf<-performance(pred,measure="acc")
plot(perf)
#经过观测取上四分位数0.468较好。

Out<-cbind(BPnet1$response,BPnet1$net.result[[1]]) #合并实际值和概率值
Out<-cbind(Out,ifelse(Out[,2]>0.468,1,0))         #以0.468为概率值分割值进行类别预测
(ConfM.BP<-table(Out[,1],Out[,3]))#计算混淆矩阵
(Err.BP<-(sum(ConfM.BP)-sum(diag(ConfM.BP)))/sum(ConfM.BP))#计算错判率。
#本例错判率为0.38预测精度不太理想,对0的预测较为准确。这与仅有3个输入变量
#且年龄不重要,也与0,1两类样本不平很有一定关系。







