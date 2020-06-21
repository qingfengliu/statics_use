#nnet包中的nnet函数可以实现传统B-P反向传播网络分类和回归预测。
#网络拓扑为二层或三层网络结构。输入节点个数等于输入变量个数。隐层只有1层
#隐节点个数由用户指定。
#二分类和回归问题的输出节点只有一个。多分类节点个数为输出变量的类别数
#nnet(输出变量~输入变量,data=数据框名,size=隐节点个数,linout=false/true,
#entropy=false/true,maxit=100,abstol=0.0001)
#size指出隐节点个数。
#linout用于指定输出节点的激活函数是否为非线性函数,false表示非线性函数。
#entropy用于指定损失函数是否采用交互熵,默认false表示损失函数采用误差平方和的形式。
#maxit迭代停止次数条件。默认为100次
#abstol也是迭代停止条件。表示权重最大调整量小于指定值
library("nnet")
BuyOrNot<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\消费决策数据.txt",header=TRUE)
set.seed(1000)
#隐节点为2.
(BPnet2<-nnet(Purchase~Age+Gender+Income,data=BuyOrNot,size=2,entropy=TRUE,abstol=0.01))
#运行有11个连接,迭代终止时的损失函数值为285.32

#predict可对nnet对象进行预测,默认以0.5作为分割值,type="class"指定输出为分类预测
#nnet可对多芬恩磊进行预测。且预测值是取各类别的概率值,最终的预测结果对应最高概率值所
#对应的类别。若不定义为因子,则按回归预测处理。
predict(BPnet2,BuyOrNot,type="class")

#使用neuralnet实现同等功能
library("neuralnet")
set.seed(1000)
(BPnet3<-neuralnet(Purchase~Age+Gender+Income,data=BuyOrNot,
                   algorithm="backprop",learningrate=0.01,hidden=2,err.fct="ce",linear.output=FALSE))