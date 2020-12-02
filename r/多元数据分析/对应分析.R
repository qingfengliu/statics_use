library(MASS)
library(ca)
data<-read.csv('D:/书籍资料整理/多元统计分析/例7-1.csv',encoding='UTF-8')
b<-table(data$rank,data$type) 
c<-as.list.data.frame(b)
c<-as.data.frame(c)
options(digits=3) #保留小数位数
temp=ca(c)
#Mass 边缘概率
#ChiDist 卡方检验结果
#Inertia  惯量
#Dim. 1 
#Dim. 2   两个因子对行、列变量的因子载荷

temp$rowcoord
temp$colcoord
plot(temp)
#对于这个包看坐标就输了