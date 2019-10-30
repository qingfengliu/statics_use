load("D:\\书籍资料整理\\统计R-贾俊平\\example9_1.RData")
library(car)
scatterplot(销售收入~广告支出,data=example9_1,pch=19,xlab="广告支出",ylab="销售收入",cex.lab=0.8)

#相关系数
load("D:\\书籍资料整理\\统计R-贾俊平\\example9_1.RData")
cor(example9_1[,2],example9_1[,3])
#检验相关系数
cor.test(example9_1[,2],example9_1[,3])

#回归拟合
model<-lm(销售收入~广告支出,data=example9_1)
summary(model)
#回归系数的置信区间
confint(model,level=0.95)
#输出方差分析表
anova(model)

#绘制拟合图
attach(example9_1)
model<-lm(销售收入~广告支出)
plot(销售收入~广告支出)
text(销售收入~广告支出,labels=企业编号,cex=0.6,adj=c(-0.6,0.25),col=4)
abline(model,col=2,lwd=2)
n=nrow(example9_1)
for (i in 1:n) {
  segments(example9_1[i,3],example9_1[i,2],example9_1[i,3],model$fitted[i])
}
mtext(expression(hat(y)==2343.8916+5.6735%*%广告支出),cex=0.7,side=1,line=-6,adj=0.75)
arrows(600,4900,550,5350,code=2,angle = 15,length = 0.08)
#######书中没有拟合优度部分可以试着补齐#########################################

#置信区间与预测区间
load("D:\\书籍资料整理\\统计R-贾俊平\\example9_1.RData")
model<-lm(销售收入~广告支出,data=example9_1)
x0<-example9_1$广告支出
pre_model<-predict(model)
con_int<-predict(model,data.frame(广告支出=x0),interval="confidence",level=0.95)
pre_init<-predict(model,data.frame(广告支出=x0),interval="prediction",level=0.95)
data.frame(销售收入=example9_1$销售收入,点预测值=pre_model,置信下限=con_int[,2],置信上限=con_int[,3],预测下限=pre_init[,2],预测上限=pre_init[,3])

#绘制置信区间和预测区间图
model<-lm(销售收入~广告支出,data=example9_1)
x0<-seq(min(example9_1$广告支出),max(example9_1$广告支出))
con_int<-predict(model,data.frame(广告支出=x0),interval="confidence",level=0.95)
pre_init<-predict(model,data.frame(广告支出=x0),interval="prediction",level=0.95)
par(cex=0.8,mai=c(0.7,0.7,0.1,0.1))
n=nrow(example9_1)
plot(销售收入~广告支出,data=example9_1)
abline(model,lwd=2)
for (i in 1:n) {
  segments(example9_1[i,3],example9_1[i,2],example9_1[i,3],model$fitted[i])
}
lines(x0,con_int[,2],lty=5,lwd=2,col="blue")
lines(x0,con_int[,3],lty=5,lwd=2,col="blue")
lines(x0,pre_init[,2],lty=5,lwd=2,col="red")
lines(x0,pre_init[,3],lty=5,lwd=2,col="red")
legend(x="topleft",legend = c("回归线","置信区间","预测区间"),lty=c(1,5,6),col=c(1,4,2),lwd=2,cex=0.8)

#计算x0=500时销售收入的点预测值,置信区间和预测区间(新值预测)
x0<-data.frame(广告支出=500)
predict(model,newdata = x0)
predict(model,data.frame(广告支出=500),interval="confidence",level=0.95)
predict(model,data.frame(广告支出=500),interval="prediction",level=0.95)

#####################回归模型的诊断判断,这节主要使用画图法#########################
#残差和标准化残差
load("D:\\书籍资料整理\\统计R-贾俊平\\example9_1.RData")
model<-lm(销售收入~广告支出,data=example9_1)
pre<-fitted(model)
res<-residuals(model)
zre<-model$residuals/(sqrt(deviance(model)/df.residual(model)))
data.frame(销售收入=example9_1$销售收入,点预测值=pre,残差=res,标准化残差=zre)

#成分残差图
load("D:\\书籍资料整理\\统计R-贾俊平\\example9_1.RData")
model<-lm(销售收入~广告支出,data=example9_1)
library(car)
crPlots(model)

#检验正态性,plot会得到很多图，其中有QQ图,残差图
par(mfrow=c(2,2),cex=0.8,cex.main=0.7)
plot(model)

#检验方差齐性，上边plot中的scale-location图可以。还可以使用散布-水平图。或者使用ncvTest函数。
load("D:\\书籍资料整理\\统计R-贾俊平\\example9_1.RData")
model<-lm(销售收入~广告支出,data=example9_1)
library(car)
ncvTest(model)
spreadLevelPlot(model)

#检验独立性
#Durbin-Watson检验
load("D:\\书籍资料整理\\统计R-贾俊平\\example9_1.RData")
model<-lm(销售收入~广告支出,data=example9_1)
library(car)
durbinWatsonTest(model)