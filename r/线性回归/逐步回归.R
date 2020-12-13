#基于的是AIC最小方法寻找,与通常书中说的不一样
data<-read.csv("D:/书籍资料整理/应用回归分析/表5-3.csv",encoding='UTF-8')
tlm<-lm(y~x4,data=data)
tstep<-step(tlm,scope = list(lower =tlm,upper=lm(y~x1+x2+x3+x4,data=data)),direction="both", trace = TRUE)
summary(tstep)
