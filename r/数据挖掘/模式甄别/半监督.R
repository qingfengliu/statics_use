###########模拟数据异常点甄别：半监督
Data<-read.table(file="D:\\书籍资料整理\\《R语言数据挖掘(第2版)》R代码和案例数据\\模式甄别模拟数据3.txt",header=TRUE,sep=",")
par(mfrow=c(2,2))
plot(Data[,1:2],main="样本观测点的分布",xlab="x1",ylab="x2",pch=Data[,3]+1,cex=0.8)
library("DMwR")
Data[which(Data[,3]==3),3]<-NA
Data$y<-factor(Data$y)
mySelfT<-function(ModelName,TestD)
{
  Yheat<-predict(object=ModelName,newdata=TestD,type="response") 
  return(data.frame(cl=ifelse(Yheat>=0.1,1,0),pheat=Yheat))
}
 
SemiP<-predict(object=SemiT,newdata=Data,type="response")
#SelfTrain(R公式,data=数据框名,
#learner(“分类函数名”,predFunc="预测函数名",thrConf=0.9,maxIts=10,percFull=比例阈值)),
#thrConf指定预测置信度阈值,高于此值进入数据集Di中，如果再没有数据集进入Di迭代结束
#maxIts迭代最大次数
#percFull当Di/D大于指定阈值时迭代结束。
SemiT<-SelfTrain(y~.,data=Data,
                 learner("glm",list(family=binomial(link="logit"))),
                 predFunc="mySelfT",thrConf=0.02,maxIts=100,percFull=1)   
SemiP<-predict(object=SemiT,newdata=Data,type="response")   
Data$SemiP<-SemiP
Data.Sort<-Data[order(x=Data$SemiP,decreasing=TRUE),]
P<-0.30
N<-length(Data[,1])
NoiseP<-head(Data.Sort,trunc(N*P))
colP<-ifelse(1:N %in% rownames(NoiseP),2,1)
a<-as.integer(as.vector(Data[,3]))
plot(Data[,1:2],main="自训练模式甄别结果(30%)",xlab="x1",ylab="x2",pch=ifelse(is.na(a),3,a)+1,cex=0.8,col=colP)
#几乎覆盖了所有已知模式,但是决策精度提升不明显。所有本例采用朴素贝叶斯分类法更理想