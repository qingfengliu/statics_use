###########################一个总体均值的检验#########################
#大样本的z检验
#alternative是选择左侧还是右侧的参数，less是左侧
load("D:\\study\\data\\example6_3.RData")
library(BSDA)
print(z.test(example6_3$PM2.5值,mu=81,sigma.x=sd(example6_3$PM2.5值),alternative="less",conf.level=0.95))
#小样本t检验
load("D:\\study\\data\\example6_4.RData")
t.test(example6_4$厚度,mu=5)


##########################两个总体均值的检验##########################
#检验的是两个总体均值的差异
load("D:\\study\\data\\example6_5.RData")
z.test(example6_5$男生上网时间,example6_5$女生上网时间,sigma.x=sd(example6_5$男生上网时间),sigma.y=sd(example6_5$女生上网时间),alternative="two.sided")

##########################独立小样本,分为3中情形.############
#总体方差已知。无论样本量大小都用z检验
#总体方差未知且不等和总体方差未知且相等使用的都是t检验。

#假设总体方差相等
load("D:\\study\\data\\example6_6.RData")
t.test(example6_6$甲企业,example6_6$乙企业,var.equal=TRUE)

#假设总体方差不等
t.test(example6_6$甲企业,example6_6$乙企业,var.equal=FALSE)

#配对数据
load("D:\\study\\data\\example6_7.RData")
t.test(example6_7$旧款饮料,example6_7$新款饮料,paired=TRUE)

#配对t检验的效应量
library(lsr)
cohensD(example6_7$旧款饮料,example6_7$新款饮料,method="paired")

#总体比例的检验，采用的是手动计算方法。
#单样本
n<-2000
p<-450/2000
pi0<-0.25
z<-(p-pi0)/sqrt(pi0*(1-pi0)/n)
p_value<-1-pnorm(z)
data.frame(z,p_value)

###################总体方差的检验##################
#一个总体方差的检验
load("D:\\study\\data\\example6_11.RData")
library(TeachingDemos)
sigma.test(example6_11$填装量,sigmasq=16,alternative="greater")

#两个总体方差比检验
load("D:\\study\\data\\example6_6.RData")
var.test(example6_6[,1],example6_6[,2],alternative = "two.sided")

##################非参数检验，只记录正态性检验##################
#Q-Q图
load("D:\\study\\data\\example6_3.RData")
par(mfrow=c(1,2),mai=c(0.7,0.7,0.2,0.1),cex=0.8)
qqnorm(example6_3$PM2.5值,xlab="期望正态值",ylab="观测值",datax=TRUE,main="正态Q-Q图")
qqline(example6_3$PM2.5值,datax=TRUE,col="red")
#P-P图
f<-ecdf(example6_3$PM2.5值)
p1<-f(example6_3$PM2.5值)
p2<-pnorm(example6_3$PM2.5值,mean(example6_3$PM2.5值),sd(example6_3$PM2.5值))
plot(p1,p2,xlab = "观测值的累积概率",ylab = "期望的累积概率",main="正态P-P图")

##shapiro-wilk正态检验
load("D:\\study\\data\\example6_4.RData")
shapiro.test(example6_4$厚度)

##K-S检验
load("D:\\study\\data\\example6_4.RData")
ks.test(example6_4$厚度,"pnorm",mean(example6_4$厚度),sd(example6_4$厚度))
#会报错Kolmogorov - Smirnov检验里不应该有连结网上搜了一种方法但是结果会有些不同。
#还可以用jitter给重复数据加噪音ks检验是计算数据分布函数与假设总体分布函数之间的差异。
#采用秩统计量，在排序过程中若有重复值的话就会显示警告
#ks.test(jitter(example6_4$厚度),"pnorm",mean(example6_4$厚度),sd(example6_4$厚度))
#这两个检验书中说十分敏感,对于对正态性要求相对宽松时慎用