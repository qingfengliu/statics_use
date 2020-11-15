#unit5

#纱年产量
a<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file4.csv",sep=",",header = T)
x<-ts(a$output,start = 1964)
#差分运算
x.dif<-diff(x)
#差分图
plot(x.dif)

#北京市居民车辆拥有
#读入数据，并绘制时序图
b<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file15.csv",sep=",",header = T)
x<-ts(b$vehicle,start = 1950)
plot(x)
#1阶差分，并绘制出差分后序列的时序图
x.dif<-diff(x)
plot(x.dif)
#2阶差分，并绘制出差分后序列的时序图
x.dif2<-diff(x,1,2)
plot(x.dif2)

######这里应该是一个季节序列
#读入数据，并绘制时序图
c<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file16.csv",sep=",",header = T)
x<-ts(c$export,start = c(2001,1),frequency = 12)
plot(x)
#1阶差分，并绘制出差分后序列的时序图
x.dif<-diff(x)
plot(x.dif)
#1阶差分加12步差分，并绘制出差分后序列的时序图
x.dif1_12<-diff(diff(x),12)
plot(x.dif1_12)
x<-arima.sim(n=1000,list(order=c(0,1,0)),sd=10)
plot(x)

#例5-5
x<-arima.sim(n=1000,list(order=c(0,1,0)),sd=10)
plot(x)


#中国农业实际国民收入指数序列。 ARIMA
#读入数据，并绘制时序图
d<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file17.csv",sep=",",header = T)
x<-ts(d$index,start = 1952)
plot(x)
#1阶差分，并绘制出差分后序列的时序图
x.dif<-diff(x)
plot(x.dif)
#绘制差分后序列自相关图和偏自相关图
acf(x.dif)
pacf(x.dif)
#拟合ARIMA(0,1,1)模型
x.fit<-arima(x,order=c(0,1,1))
x.fit
#残差白噪声检验
for(i in 1:2) print(Box.test(x.fit$residual,lag=6*i))

#中国农业实际国民收入指数序列。 ARIMA预测
library(zoo)
library(forecast)
d<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file17.csv",sep=",",header = T)
x<-ts(d$index,start = 1952)
x.fit<-arima(x,order=c(0,1,1))
x.fore<-forecast(x.fit,h=10)
plot(x.fore)

#美国23岁妇女生育率
#读入数据，并绘制时序图
e<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file18.csv",sep=",",header = T)
x<-ts(e$fertility,start = 1917)
plot(x)
#1阶差分，并绘制出差分后序列的时序图
x.dif<-diff(x)
plot(x.dif)
#绘制差分后序列自相关图和偏自相关图
acf(x.dif)
pacf(x.dif)
#拟合疏系数模型ARIMA((1,4),1,0)
x.fit<-arima(x,order = c(4,1,0),transform.pars = F,fixed = c(NA,0,0,NA))
x.fit
#残差白噪声检验
for(i in 1:2) print(Box.test(x.fit$residual,lag=6*i))
#做5期预测，并绘制预测图
x.fore<-forecast(x.fit,h=5)
x.fore
plot(x.fore)

#德国工人季节失业率(简单季节模型)
#读入数据，并绘制时序图
f<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file19.csv",sep=",",header = T)
x<-ts(f$unemployment_rate,start = c(1962,1),frequency = 4)
plot(x)
#1阶4步差分，并绘制出差分后序列的时序图
x.dif<-diff(diff(x),4)
plot(x.dif)
#绘制差分后序列自相关图和偏自相关图
acf(x.dif)
pacf(x.dif)
#拟合加法季节模型ARIMA((1,4),(1,4),0)
x.fit<-arima(x,order = c(4,1,0),seasonal = list(order=c(0,1,0),period=4,transform.par=F,fixed = c(NA,0,0,NA)))
x.fit
#残差白噪声检验
for(i in 1:2) print(Box.test(x.fit$residual,lag=6*i))
#做3年期预测，并绘制预测图
x.fore<-forecast(x.fit,h=12)
x.fore
plot(x.fore)

#美国女性月度失业率
#读入数据，并绘制时序图
g<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file20.csv",sep=",",header = T)
x<-ts(g$unemployment_rate,start = c(1948,1),frequency = 12)
plot(x)
#作1阶12步差分，并绘制出差分后序列的时序图
x.dif<-diff(diff(x),12)
plot(x.dif)
#绘制差分后序列自相关图和偏自相关图
acf(x.dif)
pacf(x.dif)
#拟合ARIMA(1,(1,12),1)模型
x.fit<-arima(x,order = c(1,1,1),seasonal = list(order=c(0,1,0),period=12))
for(i in 1:2) print(Box.test(x.fit$residual,lag=6*i))
#拟合ARIMA(1,1,1)*ARIMA(0,1,1)12模型
x.fit<-arima(x,order = c(1,1,1),seasonal = list(order=c(0,1,1),period=12))
x.fit
#残差序列白噪声检验
for(i in 1:2) print(Box.test(x.fit$residual,lag=6*i))

x.fit<-arima(x,order = c(1,1,1),seasonal = list(order=c(0,1,1),period=12))
x.fit
#残差序列白噪声检验
for(i in 1:2) print(Box.test(x.fit$residual,lag=6*i))

#残差自回归估计中国农业实际国民收入指数序列
#拟合关于时间t的线性回归模型
d<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file17.csv",sep=",",header = T)
x<-ts(d$index,start = 1952)
t<-c(1:37)

#求线性回归 x=at+b  
x.fit1<-lm(x~t)
summary(x.fit1)
#拟合关于延迟变量的自回归模型：书中
#尝试构造了两种模型,一种是变量为时间t的幂函数
#另一种是变量为延迟序列值xt-1
xlag<-x[2:37]
x2<-x[1:36]
x.fit2<-lm(x2~xlag)
summary(x.fit2)
#两个趋势拟合模型的拟合效果图
fit1<-ts(x.fit1$fitted.value,start = 1952)
fit2<-ts(x.fit2$fitted.value,start = 1952)
plot(x,type = "p",pch=8)
lines(fit1,col=2)
lines(fit2,col=4)

#残差自相关检验
library(lmtest)
#DW检验
dwtest(x.fit1)
#DWh检验
dwtest(x.fit2,order.by=xlag)

#例5-6 续(5)
#绘制差分后序列自相关图和偏自相关图
acf(x.fit1$residual)
pacf(x.fit1$residual)
#拟合AR(2)模型
r.fit<-arima(x.fit1$residual,order=c(2,0,0),include.mean = F)
r.fit
#残差自相关模型的显著性检验
for(i in 1:2) print(Box.test(r.fit$residual,lag=6*i))

#例5-11
#读入数据，并绘制时序图
h<-read.table("E:/R/data/file21.csv",sep=",",header = T)
x<-ts(h$yield_rate,start = c(1963,4),frequency = 12)
plot(x)
#作1阶差分，并绘制出差分后残差序列时序图
x.dif<-diff(x)
plot(x.dif)
#绘制1阶差分后残差平方图
plot(x.dif^2)

#例5-11 续
#作对数变换，并绘制对数变换后时序图
lnx<-log(x)
plot(lnx)
#作1阶差分，并绘制出差分后序列时序图
dif.lnx<-diff(lnx)
plot(dif.lnx)
#残差白噪声检验
for(i in 1:2) print(Box.test(dif.lnx,lag=6*i))


