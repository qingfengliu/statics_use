#例3-11
#读入数据，并绘制时序图
b<-read.table("D:\\书籍资料整理\\时间序列分析_王燕\\file10.csv",sep=",",header = T)
dif_x<-ts(diff(b$change_temp),start = 1880)
plot(dif_x)
#白噪声检验.拒绝原假设,非白噪声序列。
for(i in 1:2) print(Box.test(dif_x,type = "Ljung-Box",lag=6*i))
#绘制自相关图和偏自相关图
acf(dif_x)
pacf(dif_x)

library(zoo)
library(forecast)
#例3-11系统自动定阶ARIMA(1,0,1)  
#第一个参数是p自回归,
#第二个参数是差分次数目前讲的差分次数都是0,
#第三个参数是移动自平均次数。
#自动定阶
auto.arima(dif_x)

#手动拟合
dif_x.fit<-arima(dif_x,order = c(1,0,1))
dif_x.fit
#CSS表示最小二乘法
dif_x.fit<-arima(dif_x,order = c(1,0,1),method = "CSS")
#LB检验，非拒绝原假设
for(i in 1:2) print(Box.test(dif_x.fit$residual,lag=6*i))

#