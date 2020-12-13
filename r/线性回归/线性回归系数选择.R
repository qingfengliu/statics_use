
library("leaps")
data3.1<-read.csv("D:/书籍资料整理/应用回归分析/表3-1.csv",encoding='UTF-8')
exps<-regsubsets(y~x1+x2+x3+x4+x5+x6+x7+x8+x9,data = data3.1)
expres<-summary(exps)
#使用调整决定系数准则选取
res<-data.frame(expres$outmat,adjr2=expres$adjr2)
res

#使用Cp准侧选取
res<-data.frame(expres$outmat,Cp=expres$cp)
res

#使用BIC
res<-data.frame(expres$outmat,BIC=expres$bic)
res
