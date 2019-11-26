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