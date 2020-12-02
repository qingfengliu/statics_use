library(foreign)
library(cluster)
mydata<-read.csv('D:/书籍资料整理/多元统计分析/表3-7.csv',encoding='UTF-8')
X<-as.data.frame(mydata)
Z<-data.frame(scale(X[,2:7]),row.names = X[,1])
fresult<-fanny(Z,3,tol=0.0001,maxit=1000)
summary(fresult)
plot(fresult)

