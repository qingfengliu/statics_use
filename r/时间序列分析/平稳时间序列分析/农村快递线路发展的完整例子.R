#例1。农村投递线路新增主题数量
#读入数据，并绘制时序图
a<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file8.csv",sep=",",header = T)
x<-ts(a$kilometer,start = 1950)
plot(x)
#白噪声检验
for(i in 1:2) print(Box.test(x,type = "Ljung-Box",lag=6*i))
#绘制自相关图和偏自相关图
acf(x)
pacf(x)
library(zoo)
library(forecast)
#例3-9系统自动定阶。并会拟合s.e.标准误
auto.arima(x)

#ML 极大似然拟合
x.fit<-arima(x,order = c(2,0,0),method = "ML")
x.fit

#LB拟合检验。检验残差序列是否残留相关信息。
#如果拒绝原假设说明残差序列含有相关信息,拟合模型不显著。
for(i in 1:2) print(Box.test(x.fit$residual,lag=6*i))

#t参数检验,参数/标准误，仿佛使用最小二乘法估计的t检验不相同。df=n-m   m为参数个数
#ar1系数显著性检验
t1<-0.7185/0.1083
pt(t1,df=56,lower.tail = F)
#ar2系数显著性检验
t2<-0.5294/0.1067
pt(t2,df=56,lower.tail = T)
#ar3系数显著性检验
t0=11.0223/3.0906
pt(t0,df=56,lower.tail = F)

#预测
x.fore<-forecast(x.fit,h=5)
x.fore
plot(x.fore)

L1<-x.fore$fitted-1.96*sqrt(x.fit$sigma2)
U1<-x.fore$fitted+1.96*sqrt(x.fit$sigma2)
L2<-ts(x.fore$lower[,2],start = 2009)
U2<-ts(x.fore$upper[,2],start = 2009)
c1<-min(x,L1,L2)
c2<-max(x,L2,U2)
plot(x,type = "p",pch=8,xlim = c(1950,2013),ylim = c(c1,c2))
lines(x.fore$fitted,col=2,lwd=2)
lines(x.fore$mean,col=2,lwd=2)
lines(L1,col=4,lty=2)
lines(L2,col=4,lty=2)
lines(U1,col=4,lty=2)
lines(U2,col=4,lty=2)
