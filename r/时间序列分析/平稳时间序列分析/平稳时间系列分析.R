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





