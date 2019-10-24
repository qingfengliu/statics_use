#大数据量。总体分布无假设。 Z检验
load("D:\\书籍资料整理\\统计R-贾俊平\\example5_1.RData")
library(BSDA)
z.test(example5_1$耗油量,mu=0,sigma.x=sd(example5_1$耗油量),conf.level=0.9)

#小数据量，总体分布假设为正态并且总体方差未知。t检验
load("D:\\书籍资料整理\\统计R-贾俊平\\example5_2.RData")
t.test(example5_2,conf.level = 0.95)

#两总体均值之差。独立大样本
load("D:\\书籍资料整理\\统计R-贾俊平\\example5_3.RData")
library(BSDA)
z.test(example5_3$男性工资,example5_3$女性工资,mu=0,sigma.x=sd(example5_3$男性工资),sigma.y=sd(example5_3$女性工资),conf.level=0.95)$conf.int

#两总体均值之差。独立小样本,两样本均为小样本正态，方差未知且相等。
load("D:\\书籍资料整理\\统计R-贾俊平\\example5_4.RData")
t.test(x=example5_4$方法一,y=example5_4$方法二,var.equal=TRUE)
#方差不等
t.test(x=example5_4$方法一,y=example5_4$方法二,var.equal=FALSE)
#配对数据,小样本，两个总体正态
load("D:\\书籍资料整理\\统计R-贾俊平\\example5_5.RData")
t.test(x=example5_5$试卷A,y=example5_5$试卷B,paired = TRUE)


#总体比例的区间估计
#一个总体的比例估计,公式法
n<-500
x<-325
p<-x/n
q<-qnorm(0.975)
LCI<-p-q*sqrt(p*(1-p)/n)
UCI<-p-q*sqrt(p*(1-p)/n)
data.frame(LCI,UCI)
#或使用Hmisc包得到三种区间
#第一个区间是F检验的精确区间
#第二个区间是得分检验的wilson近似区间
#第三种是按公式计算的区间
library(Hmisc)
n<-500
x<-325
binconf(x,n,alpha=0.05,method="all")

#两个总体比例之差的估计。都是使用公式计算的。公式记忆的关键p*(1-p)

#总体方差的区间估计.
#一个总体的方差,假定总体服从正态分布
load("D:\\书籍资料整理\\统计R-贾俊平\\example5_2.RData")
library(TeachingDemos)
sigma.test(example5_2$食品重量,conf.level=0.95)

#两个总体的方差比估计
load("D:\\书籍资料整理\\统计R-贾俊平\\example5_4.RData")
var.test(example5_4$方法一,example5_4$方法二,alternative="two.sided")


