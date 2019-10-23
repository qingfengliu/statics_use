#饼图
#paste是把若干个R对象链接起来，各对象以seq指定的符号间隔。
#pie创建一个饼图，x为非负的数值向量。labels设置各分区名称，radius设置半径，init.angle=90设置从12点位置开始逆时针方向绘制。
library(xlsx)
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-1物业政策.xlsx",sheetName = "Sheet1",encoding="UTF-8")
count1<-table(table1_1$社区)
name<-names(count1)
percent<-prod.table(count1)*100
lable1<-paste(name," ",percent,"%",sep = "")
par(pin=c(3,3),mai=c(0.1,0.4,0.1,0.4),cex=0.8)
pie(count1,labels = lable1,init.angle = 90)
