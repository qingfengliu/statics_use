
#############模拟线性可分下的SVM
set.seed(12345)
#模拟一个正态分布作为输入变量,均值为0方差为1
x<-matrix(rnorm(n=40*2,mean=0,sd=1),ncol=2,byrow=TRUE)
#20个-1，20个1作为输出变量
y<-c(rep(-1,20),rep(1,20))

#将输出为1的变量+1.5
x[y==1,]<-x[y==1,]+1.5
data_train<-data.frame(Fx1=x[,1],Fx2=x[,2],Fy=as.factor(y))  #生成训练样本集

x<-matrix(rnorm(n=20,mean=0,sd=1),ncol=2,byrow=TRUE)
y<-sample(x=c(-1,1),size=10,replace=TRUE)
x[y==1,]<-x[y==1,]+1.5
data_test<-data.frame(Fx1=x[,1],Fx2=x[,2],Fy=as.factor(y)) #生成测试样本集

#绘制训练集的-1，+1散点图
plot(data_train[,2:1],col=as.integer(as.vector(data_train[,3]))+2,pch=8,cex=0.7,main="训练样本集-1和+1类散点图")

library("e1071")
#支持向量机的e1071包中的svm比较常用。
#svm(formula=公式,data=数据框名,scale=TRUE/FALSE,type=支持向量机类型,kernel=核函数名,
#gamma=g,degree=d,cost=C,epsilon=0.1,na.action=na.omit/na.fail)
#scale表示建模之前是否要标准化处理。
#type用于指定支持向量机的类型,可取值有"C-classification","eps-regression"等,分别
#表示支持向量分类C-SVM和以ε-不敏感损失函数。
#kernel用于指定核函数名,可取值有"linear","polynormial","radial basis"等。
#分别对应,线性核、多项式核、镜像核
#gamma用于指定多项式核以及镜像核的参数γ。R默认gamma是线性核中的常数项等于1/p
#degree用于指定多项式核中的阶数d。
#cost指定损失惩罚参数C。
#eplision用于指定支持向量回归中的ε-带，默认为0.1。
#na.action取na.omit表示忽略带有缺失的观测。

SvmFit<-svm(Fy~.,data=data_train,type="C-classification",kernel="linear",cost=10,scale=FALSE)
summary(SvmFit)
#svm函数的返回结果包含多个成分,主要有:
#SV:给出各支持向量在所有变量上的取值。
#index:给出各支持向量的预测编号。
#decision.values:将各观测带入决策函数给出决策函数值。依据决策函数值的正负,预测各观测所属的类别

SvmFit$index
plot(x=SvmFit,data=data_train,formula=Fx1~Fx2,svSymbol="#",dataSymbol="*",grid=100)
#本例包含40个观测。由分布图可见该问题是一个广义线性可分支持下的分类问题，可采用线性核函数
#设置惩罚系数为10，找到16个支持向量(我做试验的结果是找到了15个一类8一类7)
#使用plot作出可视化最大边界超平面和支持向量
#svsymbol指出支持向量的符号。dataSymbol是一般变量的符号

#cost为0.1时,因惩罚系数降低,厚度增大找到支持向量25个
SvmFit<-svm(Fy~.,data=data_train,type="C-classification",kernel="linear",cost=0.1,scale=FALSE)
summary(SvmFit)



##############10折交叉验证选取损失惩罚参数C
#利用tune.svm尝试不同损失惩罚函数。10折交叉验证的预测误差估计最低是C=1
#平均判错率为0.15,标准差为0.129。该函数下的模型为最优模型。找到了17个支持向量。
#该参数下的模型为最优模型,找到了17个支持向量。利用该模型对测试样本集中的10个观测类别
#预测,预测错误率为0.2。我做的实现最佳为cost=0.1错误率为0.15。
#tune.svm(formula=公式,data=数据框名,scale=TRUE/FALSE,type=支持向量机类型,kernel=核函数名,
#gamma=g,degree=d,cost=C,epsilon=0.1,na.action=na.omit/na.fail)
#多数参数与svm相同，不同的是gamma,degree,cost为一个包含所有可能参数值的向量。
#返回值是列表对象。包括best.parameters,best.performance,best.model等成分分别对应
#存储误差最小时的参数、误差以及相应参数下的基本信息
set.seed(12345)
tObj<-tune.svm(Fy~.,data=data_train,type="C-classification",kernel="linear",
               cost=c(0.001,0.01,0.1,1,5,10,100,1000),scale=FALSE)
summary(tObj)
BestSvm<-tObj$best.model
summary(BestSvm)
yPred<-predict(BestSvm,data_test)
(ConfM<-table(yPred,data_test$Fy))
(Err<-(sum(ConfM)-sum(diag(ConfM)))/sum(ConfM))