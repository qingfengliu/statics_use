library(xlsx)
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-1物业政策.xlsx",sheetName = "Sheet1",encoding="UTF-8")
#频数分布
count1<-table(table1_1$社区)

#频数百分比
prop.table(count1)*100

#二维列联表
 mytable<-table(table1_1$社区,table1_1$性别)
 
 #将二维链表加边缘和
 addmargins(mytable)

 #二维列联表概率
 addmargins(prop.table(mytable))*100

 #使用crosstable生成二维列联表并进行百分比分析
library(gmodels)
CrossTable(table1_1$性别,table1_1$态度)
#第一个百分比为x1/(x1+x2)   ,第二个百分比为y1/(y1+y2) ,  第三个值为x1/total


#多维列联表
mytable=ftable(table1_1)
ftable(table1_1,row.vars = c("性别","态度"),col.vars = "社区")
ftable(addmargins(prop.table(table(table1_1$性别,table1_1$态度,table1_1$社区)))*100)

#数值数据生成频数分布
#breaks为要分组数,例如breaks=5表示分成五组；breaks=10*(5:10)表示将50~100之间的数据分成10组
#right=true表示包含上限值
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-1销售额.xlsx",sheetName = "Sheet1",encoding="UTF-8")
vector2<-as.vector(table1_1$销售额)
d<-table(cut(vector2,breaks = 10*(16:28),right = FALSE))
df<-data.frame(d)
percent<-df$Freq/sum(df$Freq)*100
cumsump<-cumsum(percent)
mytable<-data.frame(d,percent,cumsump)

#试做一个图barplot为创建条形图,参数概念在这个例子里不言而喻，col为设置图形颜色
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-1物业政策.xlsx",sheetName = "Sheet1",encoding="UTF-8")
count1<-table(table1_1$社区)
count2<-table(table1_1$性别)
count3<-table(table1_1$态度)
par(mfrow=c(1,3),mai=c(0.7,0.7,0.6,0.1),cex.main=0.8)
barplot(count1,xlab="频数",ylab = "社区",horiz = TRUE,main="(a)水平条形图",col=2:5)
barplot(count2,xlab="性别",ylab = "频数",main="(b)垂直条形图")
barplot(count3,xlab="态度",ylab = "频数",main="(c)垂直条形图",col=2:3)