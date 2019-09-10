#抽样与筛选
library(xlsx)
table1_2<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\1-2学生成绩50名.xlsx",sheetName = "Sheet1",encoding="UTF-8")

#无放回简单随机抽样
sample(table1_2$姓名,10,replace = FALSE)

#放回简单随机抽样
sample(table1_2$姓名,10,replace = TRUE)

#筛选出考试分数小于60的所有学生
sample(table1_2$姓名[table1_2$考试分数<60])


#筛选出考试分数大于等于90的所有分数
sample(table1_2$考试分数[table1_2$考试分数>=90])

#按性别分层，抽取20%学生做样本
library(DescTools)
d<-table1_2
n1=round(length(d$性别[d$性别=="男"])*0.2)
n2=round(length(d$性别[d$性别=="女"])*0.2)
stratified1<-Strata(table1_2,stratanames = "性别",size=c(n1,n2),method="srswor")
stratified1

#按性别和专业分层,抽取20%的学生作为样本.
#比例2,1,2,2,1,2根据比例算出调时注意顺序
stratified2=Strata(table1_2,c("性别","专业"),size=c(2,1,2,2,1,2),method = "srswor")
stratified2

#系统抽样方式抽取20%的学生作为样本
library(doBy)
systematic=sampleBy(~1,frac = 0.2,replace=FALSE,data = table1_2,systematic = T)
#~1代表数据没有分组，frac为抽取间隔 1/0.2,replace为不重复抽取，data为数据框,systematic为系统抽样
systematic


#函数的编写
myfun<-function(x){
  n<-length(x)
  mean<-sum(x)/n
  median<-median(x)
  r<-max(x)-min(x)
  s<-sd(x)
  summ<-data.frame(c(mean,median,r,s),row.names=c("平均数","中位数","极差","标准差"))
  names(summ)<-"值"
  return(summ)
}
x<-table1_2[,4]
myfun(x)

