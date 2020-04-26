library(xlsx)
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\3-130名学生成绩.xlsx",sheetName = "Sheet1",encoding="UTF-8")

#平均数
mean(table1_1$分数)


table1_2<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\3-230名学生分组数据.xlsx",sheetName = "Sheet1",encoding="UTF-8")

#加权平均数
weighted.mean(table1_2$组中值,table1_2$人数)

#中位数
median(table1_1$分数)

#分位数
quantile(table1_1$分数,probs = c(0.25,0.75),type=7)

#百分位数
quantile(table1_1$分数,probs = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),type=6)

#众数R无函数，对于众数我想说仅存在于理论上，实际很少使用

#极差
rang<-max(table1_1$分数)-min(table1_1$分数);rang

#四分位差
IQR(table1_1$分数,type = 6)

#方差
var(table1_1$分数)

#标准差
sd(table1_1$分数)

#变异系数 sd/mean


#标准分数数组 ？
as.vector(round(scale(table1_1$分数),4))

#偏度系数
library(agricolae)
skewness(table1_1$分数)

#峰度系数
kurtosis(table1_1$分数)

#输出多个描述统计量
table1_3<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\3-3气手枪.xlsx",sheetName = "Sheet1",encoding="UTF-8")

library(pastecs)
round(stat.desc(table1_3),4)

library(psych)
describe(table1_3)