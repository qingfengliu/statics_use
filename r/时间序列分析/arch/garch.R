
library(FinTS)
library(tseries)
#例5-12
#读入数据，并绘制时序图
k<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file22.csv",sep=",",header = T)
x<-ts(k$returns,start = c(1926,1),frequency = 12)
plot(x)
#绘制序列平方图
plot(x^2)

#例5-12 续
#LM检验

for(i in 1:5) print(ArchTest(x,lag=i))
#Portmanteau Q检验
for(i in 1:5) print(Box.test(x^2,lag=i))
#拟合ARCH(3)模型
x.fit<-garch(x,order=c(0,3))
summary(x.fit)
#绘制条件异方差模型拟合的95%置信区间
x.pred<-predict(x.fit)
plot(x.pred)
#条件异方差置信区间和方差齐性置信区间比较图示
plot(x)
lines(x.pred[,1],col=2)
lines(x.pred[,2],col=2)
abline(h=1.96*sd(x),col=4,lty=2)
abline(h=-1.96*sd(x),col=4,lty=2)
#5.6.2 GARCH模型
#例5-13
#读入数据，并绘制时序图
w<-read.table("E:/R/data/file23.csv",sep=",",header = T)
x<-ts(w$exchange_rates,start=c(1979,12,31),frequency = 365)
plot(x)
#对差分序列性质考察
plot(diff(x))
acf(diff(x))
pacf(diff(x))
#水平相关信息提取，拟合ARIMA(0,1,1)模型
x.fit<-arima(x,order = c(0,1,1))
x.fit
#残差白噪声检验
for (i in 1:6) print(Box.test(x.fit$residual,type = "Ljung-Box",lag=i))
#水平预测，并绘制预测图
x.fore<-forecast(x.fit,h=365)
plot(x.fore)
#条件异方差检验(Portmanteau Q检验)
for (i in 1:6) print(Box.test(x.fit$residual^2,type = "Ljung-Box",lag=i))
#拟合GARCH(1,1)模型
r.fit<-garch(x.fit$residual,order=c(1,1))
summary(r.fit)
#绘制波动置信区间
r.pred<-predict(r.fit)
plot(r.pred)