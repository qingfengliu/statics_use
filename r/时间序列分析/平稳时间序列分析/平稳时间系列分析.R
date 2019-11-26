#unit3

#例3-1
x1<-arima.sim(n=100,list(ar=0.8))
x3<-arima.sim(n=100,list(ar=c(1,-0.5)))
e<-rnorm(100)
x2<-filter(e,filter = -1.1,method = "recursive")
x4<-filter(e,filter = c(1,0.5),method = "recursive")
ts.plot(x1)
ts.plot(x2)
ts.plot(x3)
ts.plot(x4)

#例3-5
#arima.sim生产ARMA模型拟合数据,ar=0.8代表,拟合的是ar模型,一阶系数为0.8
#系数按阶数降序排列
#acf自相关图,pacf偏自相关图
x1<-arima.sim(n=1000,list(ar=0.8))
x2<-arima.sim(n=1000,list(ar=-0.8))
x3<-arima.sim(n=1000,list(ar=c(1,-0.5)))
x4<-arima.sim(n=1000,list(ar=c(-1,-0.5)))
acf(x1)
acf(x2)
acf(x3)
acf(x4)

#例3-5 续
pacf(x1)
pacf(x2)
pacf(x3)
pacf(x4)

#例3-6
x1<-arima.sim(n=1000,list(ma=-2))
x2<-arima.sim(n=1000,list(ma=-0.5))
x3<-arima.sim(n=1000,list(ma=c(-4/5,16/25)))
x4<-arima.sim(n=1000,list(ar=c(-5/4,25/16)))
acf(x1)
acf(x2)
acf(x3)
acf(x4)

#例3-6 续(2)
pacf(x1)
pacf(x2)
pacf(x3)
pacf(x4)

#例3-8
x<-arima.sim(n=1000,list(ar=0.5,ma=-0.8))
acf(x)
pacf(x)

#例3-9
#读入数据，并绘制时序图
a<-read.table("E:/R/data/file8.csv",sep=",",header = T)
x<-ts(a$kilometer,start = 1950)
plot(x)
#白噪声检验
for(i in 1:2) print(Box.test(x,type = "Ljung-Box",lag=6*i))
#绘制自相关图和偏自相关图
acf(x)
pacf(x)

#例3-10
#读入数据，并绘制时序图
overshort<-read.table("E:/R/data/file9.csv",sep=",",header = T)
overshort<-ts(overshort)
plot(overshort)
#白噪声检验
for(i in 1:2) print(Box.test(overshort,type = "Ljung-Box",lag=6*i))
#绘制自相关图和偏自相关图
acf(overshort)
pacf(overshort)

#例3-11
#读入数据，并绘制时序图
b<-read.table("E:/R/data/file10.csv",sep=",",header = T)
dif_x<-ts(diff(b$change_temp),start = 1880)
plot(dif_x)
#白噪声检验
for(i in 1:2) print(Box.test(dif_x,type = "Ljung-Box",lag=6*i))
#绘制自相关图和偏自相关图
acf(dif_x)
pacf(dif_x)

library(zoo)
library(forecast)

#例3-10系统自动定阶
auto.arima(overshort)
#例3-11系统自动定阶
auto.arima(dif_x)



#例3-10续(1)
overshort<-read.table("E:/R/data/file9.csv",sep=",",header = T)
overshort<-ts(overshort)
overshort.fit<-arima(overshort,order = c(0,0,1))
overshort.fit

#例3-11续(1)
b<-read.table("E:/R/data/file10.csv",sep=",",header = T)
dif_x<-ts(diff(b$chang_temp),start = 1880)
dif_x.fit<-arima(dif_x,order = c(1,0,1))
dif_x.fit




#例3-10续(2)
overshort<-read.table("E:/R/data/file9.csv",sep=",",header = T)
overshort<-ts(overshort)
overshort.fit<-arima(overshort,order = c(0,0,1))
for(i in 1:2) print(Box.test(overshort.fit$residual,lag=6*i))

#例3-11续(2)
b<-read.table("E:/R/data/file10.csv",sep=",",header = T)
dif_x<-ts(diff(b$chang_temp),start = 1880)
dif_x.fit<-arima(dif_x,order = c(1,0,1),method = "CSS")
for(i in 1:2) print(Box.test(dif_x.fit$residual,lag=6*i))


#例3-15
#读入数据，绘制时序图
x<-read.table(file = "E:/R/data/file11.csv",sep=",",header = T)
x<-ts(x)
plot(x)
#序列白噪声检验
for(i in 1:2) print(Box.test(x,lag=6*i))
#绘制自相关图和偏自相关图
acf(x)
pacf(x)
#拟合MA(2)模型
x.fit1<-arima(x,order = c(0,0,2))
x.fit1
#MA(2)模型显著性检验
for(i in 1:2) print(Box.test(x.fit1$residual,lag=6*i))
#拟合AR(1)模型
x.fit2<-arima(x,order = c(1,0,0))
x.fit2
#AR(1)模型显著性检验
for(i in 1:2) print(Box.test(x.fit2$residual,lag=6*i))


