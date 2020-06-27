#EM聚类。
##############混合高斯分布模拟
library("MASS")  
set.seed(12345)
mux1<-0    #x期望,1类
muy1<-0    #y期望,1类
mux2<-15   #x期望,2类
muy2<-15   #y期望,2类
ss1<-10    #x的方差
ss2<-10    #y的方差
s12<-3     #下，y的协方差
#生成协方差阵
sigma<-matrix(c(ss1,s12,s12,ss2),nrow=2,ncol=2)  
#生成1类随机样本,服从正态二元分布
Data1<-mvrnorm(n=100,mu=c(mux1,muy1),Sigma=sigma,empirical=TRUE)  
Data2<-mvrnorm(n=50,mu=c(mux2,muy2),Sigma=sigma,empirical=TRUE) 
#得到混合高斯分布
Data<-rbind(Data1,Data2)
plot(Data,xlab="x",ylab="y")
library("mclust")
#高斯分布的核密度估计
DataDens<-densityMclust(data=Data)      
plot(x=DataDens,type="persp",col=grey(level=0.8),xlab="x",ylab="y") 

#########################对模拟数据的EM聚类
library("mclust") 
#EM聚类
#Mclust(data=矩阵或数据框)
#返回值包含多个类别
#G:最优聚类数
#BIC:取最优聚类数目时的BIC值
#Loglik:取最优聚类数目时的对数似然值
#z:n*K的矩阵,为各观测属于各类的概率
#classification:各观测所属的小类
#uncertity:各观测不属于所属小类的概率
EMfit<-Mclust(data=Data)  
#查看聚类结果基本信息
summary(EMfit)
#显示所估计参数的估计值
summary(EMfit,parameters=TRUE)   
plot(EMfit,"classification") 
plot(EMfit,"uncertainty")
plot(EMfit,"density")

#############或者
(BIC<-mclustBIC(data=Data))
plot(BIC,G=1:7,col="black")
(BICsum<-summary(BIC,data=Data))
mclust2Dplot(Data,classification=BICsum$classification,parameters=BICsum$parameters)
