#例3-10
#读入数据，并绘制时序图
overshort<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file9.csv",sep=",",header = T)
overshort<-ts(overshort$overshort)
plot(overshort)
#白噪声检验
for(i in 1:2) print(Box.test(overshort,type = "Ljung-Box",lag=6*i))
#绘制自相关图和偏自相关图
acf(overshort)
pacf(overshort)
overshort.fit<-arima(overshort,order = c(0,0,1))
overshort.fit

#模型适合检验，残差无信息
for(i in 1:2) print(Box.test(overshort.fit$residual,lag=6*i))
library(zoo)
library(forecast)
#例3-10系统自动定阶
auto.arima(overshort)
