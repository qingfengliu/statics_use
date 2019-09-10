library(xlsx)
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\1-1学生成绩.xlsx",sheetName = "Sheet1",encoding="UTF-8")
head(table1_1,3)      #table1_1是数据框可也用R的data.frame创建
newdata1<-table1_1[order(table1_1$统计学,decreasing=TRUE),] #排序,或使用sort
mean(table1_1$统计学) #平均数
mean(table1_1[,2])
colSums(table1_1[,1:5])

#空值的处理，当存在空值时无法计算。sum带NA的行会返回NA
#sum(x,na.rm=TRUE)    #删除NA再计算
#table1_1_3<-na.omit(table1_1_2)  #可以删除table1_1_2中所有缺失值

#数据合并，可以用于数据框
#rbind(d1,d2)    #相当于union
#cbind(d3,d4[2:3])  #相当于join限制是行数必须相等

#数据类型转换
#数据框转矩阵
matrix1_1<-as.matrix(table1_1[,2:6])
rownames(matrix1_1)=table1_1[,1]
#矩阵转数据框
as.data.frame(matrix1_1)

#生成随机数
#R生成的是随机分布的随机,在分布函数之前加r
rnorm(10)    #标准正态分布
set.seed(15)  #设定随机数种子
rnorm(10,50,5) #平均数为50,方差为5
#runif均匀分布随机数，rexp指数分布随机数,rchisq卡方分布随机数
