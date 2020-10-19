#例3-15
#读入数据，绘制时序图
#ts 应该是差分的意思
x<-read.table(file = "D:\\书籍资料整理\\时间序列分析_王燕\\file11.csv",sep=",",header = T)
x<-ts(x$yield)
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

#这里MA(2) 和AR(1)在检验上都不会有什么错误。
#若要找出最适合的模型需要使用AIC和SBC准则判断。这里拟合函数arima仅给出.AIC,
#sigma^2方差。
#log likelihood。极大似然值。
#AIC=-2(极大似然函数值)+2(模型中未知参数个数)。