#条形图，简单条形图
#barplot为创建条形图,参数概念在这个例子里不言而喻，col为设置图形颜色
library(xlsx)
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-1物业政策.xlsx",sheetName = "Sheet1",encoding="UTF-8")
count1<-table(table1_1$社区)
count2<-table(table1_1$性别)
count3<-table(table1_1$态度)
par(mfrow=c(1,3),mai=c(0.7,0.7,0.6,0.1),cex.main=0.8)
barplot(count1,xlab="频数",ylab = "社区",horiz = TRUE,main="(a)水平条形图",col=2:5)
barplot(count2,xlab="性别",ylab = "频数",main="(b)垂直条形图")
barplot(count3,xlab="态度",ylab = "频数",main="(c)垂直条形图",col=2:3)

#帕累托图，是排序的条形图，并且有一条频率累计曲线

count1<-table(table1_1$社区)
par(mai=c(0.7,0.7,0.1,0.8),cex=0.8)
x<-sort(count1,decreasing=TRUE)
bar<-barplot(x,xlab = "社区",ylab="频数",ylim=c(0,1.2*max(count1)),col=2:5)
text(bar,x,labels = x,pos=3)
y<-cumsum(x)/sum(x)
par(new=T)
plot(y,type = "b",lwd=1.5,pch=15,axes=FALSE,xlab=' ',ylab=' ',main=' ')
axis(4)
mtext("累积频率",side=4,line = 3)
mtext("累积分布曲线",line=-2.5,cex=0.8,adj=0.75)

#复式条形图，当有两个类别变量时,可以将二维列联表数据绘制成复式条形图。
#复式条形图包括，并列条形图和堆叠条形图
#legend设定图例,args.legend设置图例的位置参数

mytable1<-table(table1_1$态度,table1_1$社区)
par(mfrow=c(2,2),cex=0.6)
barplot(mytable1,xlab = "社区",ylab = "频数",ylim = c(0,30),col=c("green","blue"),
        legend=rownames(mytable1),args.legend = list(x=12),beside = TRUE,main="(a)社区并列条形图")

barplot(mytable1,xlab = "社区",ylab = "频数",ylim = c(0,30),col=c("green","blue"),
        legend=rownames(mytable1),args.legend = list(x=4.5),main="(a)社区堆叠条形图")

#脊形图
library(vcd)
spine(社区~性别,data=table1_1,xlab="性别",ylab="社区",margins=c(4,3.5,1,2.5))

#马赛克图，两个以上类别的变量
mosaicplot(~性别+社区+态度,data = table1_1,color=2:3,main="")