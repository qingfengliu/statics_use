################多个变量的相关图###########
load("D:\\书籍资料整理\\统计R-贾俊平\\example10_1.RData")
library(corrgram)
corrgram(example10_1[2:7],order=TRUE,lower.panel=panel.shade,
         upper.panel=panel.pie,text.panel = panel.txt)
#蓝色和从左下角指向右上角的图表示正相关。

###回归拟合
load("D:\\书籍资料整理\\统计R-贾俊平\\example10_1.RData")
model1<-lm(y~x1+x2+x3+x4+x5,data =example10_1)
summary(model1)
#置信区间
confint(model1,level = 0.95)
#输出方差分析表
anova(model1)

###模型诊断
#残差图
par(mfrow=c(1,2),mai=c(0.8,0.8,0.4,0.1),cex=0.8,cex.main=0.7)
plot(model1,which = 1:2)
#从残差图可以看出2,4,16这几个点偏差较大。
#去掉第二个点和第四个点
newmodel1<-lm(y~x1+x2+x3+x4+x5,data = example10_1[-c(2,4),])
par(mfrow=c(1,2),mai=c(0.8,0.8,0.4,0.1),cex=0.8)
plot(newmodel1,which=1:2)

#################多重共线性及其识别##################
#
load("D:\\书籍资料整理\\统计R-贾俊平\\example10_1.RData")
#相关系数检验
library(psych)
corr.test(example10_1[3:7],use="complete")

#计算容忍度和VIF
model1<-lm(y~x1+x2+x3+x4+x5,data = example10_1)
library(car)
vif(model1)
1/vif(model1)

########################################
#逐步回归   
load("D:\\书籍资料整理\\统计R-贾俊平\\example10_1.RData")
model1<-lm(y~x1+x2+x3+x4+x5,data = example10_1)
model2<-step(model1)

#拟合逐步回归模型
model2<-lm(y~x1+x2+x5,data = example10_1)
summary(model2)

#图形诊断
par(mfrow=c(1,2),mai=c(0.8,0.8,0.4,0.1),cex=0.8)
plot(model2,which = 1:2)
#根据图中可以看出残差正态性存在疑问，该模型可能要引入些二次项。

################################################
##自变量重要性——标准化回归系数
load("D:\\书籍资料整理\\统计R-贾俊平\\example10_1.RData")
model1<-lm(y~x1+x2+x3+x4+x5,data=example10_1)
library(lm.beta)
model1.beta<-lm.beta(model1)
summary(model1.beta)

####模型比较
#书中内容与R例子并不完美匹配。书中的例子加入了二次项。

load("D:\\书籍资料整理\\统计R-贾俊平\\example10_1.RData")
model1<-lm(y~x1+x2+x3+x4+x5,data=example10_1)
model2<-lm(y~x1+x2+x5,data=example10_1)
anova(model2,model1)

##################
#计算置信区间和预测区间
load("D:\\书籍资料整理\\统计R-贾俊平\\example10_1.RData")
model2<-lm(y~x1+x2+x5,data=example10_1)
x<-example10_1[,c(3,4,7)]
pre<-predict(model2)
res<-residuals(model2)
zre<-rstandard(model2)
con_int<-predice(model2,x,interval="confidence",level=0.95)
pre_int<-predict(model2,x,interval="prediction",level=0.95)
mysummary<-data.frame(营业额=example10_1$y,点预测值=pre,残差=res,标准化残差=zre,置信下限=con_int[,2],
                         置信上限=con_int[,3],预测下线=pre_int[,2],预测上限=pre_int[,3])
round(mysummary,3)
model2<-lm(y~x1+x2+x5,data=example10_1)
x0<-data.frame(x1=50,x2=100,x5=10)
predict(model2,newdata=x0)

predice(model2,data.frame(x1=50,x2=100,x5=10),interval="confidence",level=0.95)
predict(model2,data.frame(x1=50,x2=100,x5=10),inteval="prediction",level=0.95)


###########哑变量回归#################
load("D:\\书籍资料整理\\统计R-贾俊平\\example10_7.RData")
model_s<-lm(日均营业额~用餐平均支出,data=example10_7)
summary(model_s)

#方差分析表
anova(model_s)

###哑变量回归
load("D:\\书籍资料整理\\统计R-贾俊平\\example10_7.RData")
model_dummy<-lm(日均营业额~用餐平均支出+交通方便程度,data=example10_7)
summary(model_dummy)

#方差分析表
anova(model_dummy)


##############交通方便与不方便的两个回归图像
attach(example10_7)
plot(日均营业额~用餐平均支出,pch=c(21,19)[交通方便程度],col=c('blue','red')[交通方便程度],
          main='模型:日均营业额~用餐平均支出+交通方便程度',cex.lab=0.8,cex.main=0.8)
rc<-lm(日均营业额~用餐平均支出+交通方便程度)$coef
abline(rc[1],rc[2],lty=2,col='blue',lwd=2)
abline(rc[1]+rc[3],rc[2],col='red',lwd=2)
legend(x="bottomright",legend=c('交通=方便','交通=不方便'),lty=c(1,2),col=c(2,4),lwd=2,cex=0.7)

#哑变量回归预测
model_dummy<-lm(日均营业额~用餐平均支出+交通方便程度,data=example10_7)
pre_model_dummy<-model_dummy$fitted.values
res_model_dummy<-model_dummy$residuals
mysummary<-data.frame(example10_7,点预测值=pre_model_dummy,残差=res_model_dummy)

model_s<-lm(日均营业额~用餐平均支出+交通方便程度,data=example10_7)
anova(model_s,model_dummy)
AIC(model_s,model_dummy)








