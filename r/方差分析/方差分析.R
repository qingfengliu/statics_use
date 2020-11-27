#将短格式数据转为长格式数据
load("D:\\书籍资料整理\\统计R-贾俊平\\example8_1.RData")
example8_1<-cbind(example8_1,id=factor(1:10))
library(reshape)
example8_2<-melt(example8_1,id.vars=c("id"),variable_name="品种")
example8_2<-rename(example8_2,c(id="地块",value="产量"))
save(example8_2,file="D:\\书籍资料整理\\统计R-贾俊平\\example8_2.RData")
head(example8_2)

load("D:\\书籍资料整理\\统计学\\example8_2.RData")
attach(example8_2)
boxplot(产量~品种,col="gold",main="",ylab="产量",xlab="品种")

#计算描述统计量
my_summary<-function(x){with(x,data.frame("均值"=mean(产量),"标准差"=sd(产量),n=length(产量)))}
library(plyr)
ddply(example8_2,.(品种),my_summary)

#方差分析表
attach(example8_2)
model_1w<-aov(产量~品种)   #aov提供方差分析拟合,formula用于指定模式。data为数据框,y~A y为因变量,A为因子(自变量)
summary(model_1w)

#均值图
library(gplots)
plotmeans(产量~品种,data=example8_2)

#效应量分析
library(DescTools)
model_1w<-aov(产量~品种)  #eta.sq为效应量
EtaSq(model_1w,anova = T)

#####################多重分析(显著性检验)#######################
##(1)LSD
load("D:\\书籍资料整理\\统计R-贾俊平\\example8_2.RData")
library(agricolae)
model_1w<-aov(产量~品种,data=example8_2)
LSD<-LSD.test(model_1w,"品种")
LSD          #$group中如果有相同的字母代表没显著差异
#或
library(DescTools)
PostHocTest(model_1w,method = "lsd")

##(2)HSD方法
TukeyHSD(model_1w)

#绘制配对差值置信区间比较图
plot(TukeyHSD(model_1w))

#或
library(agricolae)
HSD<-HSD.test(model_1w,"品种");HSD

##############双因子方差分析###################
table8_4<-read.csv("D:\\书籍资料整理\\统计R-贾俊平\\table8_4.csv")
table8_4<-cbind(table8_4,id=c(factor(1:10)))
library(reshape)
example8_5<-melt(table8_4,id.vars=c("id","施肥方式"))
example8_5<-rename(example8_5,c(variable="品种",value="产量"))
save(example8_5,file="D:\\书籍资料整理\\统计R-贾俊平\\example8_5.RData")
head(example8_5)
#绘制品种和施肥方式的线箱图
attach(example8_5)
boxplot(产量~品种+施肥方式,col=c("gold","green","red"),ylab="产量",xlab="品种与施肥方式")

#按品种和施肥方式交叉分类计算均值和标准差
library(reshape)
library(agricolae)
mystats<-function(x){c(n=length(x),mean=mean(x),sd=sd(x))}
dfm<-melt(example8_5,measure.vars = "产量",id.vars = c("品种","施肥方式"))
cast(dfm,品种+施肥方式+variable~.,mystats)

#主效应方差分析表
example8_5<-read.csv("D:/书籍资料整理/统计学/example8_5.csv")
model_2wm<-aov(产量~品种+施肥方式,data=example8_5)
summary(model_2wm)
model_2wm$coefficients
#主效应和交互效应图
library(HH)
interaction2wt(产量~施肥方式+品种,data=example8_5)

#效应量分析
model_2wm<-aov(产量~品种+施肥方式)
library(DescTools)
EtaSq(model_2wm,anova = T)  #eta.sq.part是偏效应。比如在排除施肥影响后，品种因子解释产生了80.69%的误差
                            #eta.sq为每个因子对这个总体的影响

#模型比较,有交互的模型与无交互的模型
model_2wm<-lm(产量~品种+施肥方式,data=example8_5)
model_2wi<-lm(产量~品种+施肥方式+品种:施肥方式,data=example8_5)
anova(model_2wm,model_2wi)
#P>显著水平,代表没有证据表明两模型有差异。侧面反映了交互效应不明显。

#####方齐性检验#######################
#方差分析的残差图和标准化残差的Q-Q图
load("D:\\书籍资料整理\\统计R-贾俊平\\example8_2.RData")
modle_1w<-aov(产量~品种,data=example8_2)
par(mfrow=c(1,2),mai=c(0.5,0.5,0.2,0.1),cex=0.6,cex.main=0.7)
plot(modle_1w,which=c(1,2))

#不同施肥方式产量的Bartlett检验
load("D:\\书籍资料整理\\统计R-贾俊平\\example8_5.RData")
bartlett.test(产量~施肥方式,data = example8_5)

#不同品种产量的Bartlett检验
load("D:\\书籍资料整理\\统计R-贾俊平\\example8_5.RData")
bartlett.test(产量~品种,data = example8_5)

#不同品种产量的Levene检验
load("D:\\书籍资料整理\\统计R-贾俊平\\example8_5.RData")
library(car)
leveneTest(产量~品种,data=example8_5)

#不同施肥方式产量的Levene检验
load("D:\\书籍资料整理\\统计R-贾俊平\\example8_5.RData")
library(car)
leveneTest(产量~施肥方式,data=example8_5)

#非参数检验 Kruskal-wallis检验
load("D:\\书籍资料整理\\统计R-贾俊平\\example8_2.RData")
attach(example8_2)
kruskal.test(产量~品种)