library(xlsx)
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-1销售额.xlsx",sheetName = "Sheet1",encoding="UTF-8")
d<-table1_1$销售额
par(mfrow=c(2,2),mai=c(0.6,0.6,0.4,0.1),cex=0.7)  #cex字体大小
hist(d,xlab = "销售额",ylab = "频数",main="(a)普通")
hist(d,breaks=20,col="lightblue",xlab = "销售额",ylab = "频数",main="(b)分成20组")
hist(d,breaks=20,prob=TRUE,col="lightblue",xlab = "销售额",ylab = "密度",main="(c)增加轴须线和核密度线")
rug(d)#R作图很怪，直到下一个图函数之前。这个图函数之后的函数都是当前图上的效果
lines(density(d),col="red")

#下一个作图函数出来了
hist(d,prob=TRUE,breaks = 20,col="lightblue",xlab = "销售额",ylab = "密度",main="(d)增加正态密度线")
curve(dnorm(x,mean(d),sd(d)),add=TRUE,col="red")
rug(jitter(d))

hist(faithful$eruptions,breaks=20,probability = TRUE,xlab = "喷发持续时间",col="lightblue",main="")
rug(faithful$eruptions)
lines(density(faithful$eruptions,bw=0.1),lwd=2,col='red')
curve(dnorm(x,mean = mean(faithful$eruptions),sd=sd(faithful$eruptions)),add=TRUE,col="blue",lwd=2,lty=6)
#########################以上绘制的是直方图#############################

####书中例子茎叶图就不画了，因为众数评估的很少


############线箱图#################
#range=1.5,1.5倍的四分位差为内围栏,=3为外围栏(全包括)貌似与range=0极值一样的。
#varwidth=TRUE,箱子宽度与样本量有关。
par(mfrow=c(3,1),mai=c(0.4,0.2,0.3,0.2))
x<-rnorm(1000,50,5)
boxplot(x,range = 1.5,col="red",horizontal = TRUE,main="相邻值与箱子连线的线箱图range=1.5",cex=0.8)
boxplot(x,range = 3,col="green",horizontal = TRUE,main="相邻值与箱子连线的线箱图range=3",cex=0.8)
boxplot(x,range = 0,varwidth = TRUE,col="pink",horizontal = TRUE,main="极值与箱子连线的箱线图(range=0,varwidth=T)",cex=0.8)

#不同分布的线箱图
x<-table1_1$销售额
layout(matrix(c(1,2),nc=1),heights = c(2,1))
par(mai=c(0.15,0.4,0.2,0.2),cex=0.8)
hist(x,freq = FALSE,col = "lightblue",breaks = 15,xlab="",ylab = "",main="")
rug(x,col="blue4")
abline(v=quantile(x),col="blue4",lwd=2,lty=6)
points(quantile(x),c(0,0,0,0,0),lwd=5,col="red2")
lines(density(x),col="red",lwd=2)
par(mai=c(0.35,0.42,0.2,0.43),cex=0.8)
boxplot(x,col="pink",lwd=2,horizontal = TRUE)
rug(x,ticksize = 0.1,col="blue4")
abline(v=quantile(x),col="blue4",lwd=2,lty=6)


table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-3打手枪.xlsx",sheetName = "Sheet1",encoding="UTF-8")
boxplot(table1_1,col = "lightblue",xlab="运动员",ylab="射击环数",cex.lab=0.8,cex.axis=0.6)

#小提琴图，太奇葩了没见过谁用
#点图,
example2_3<-cbind(table1_1,id=factor(1:20))
library(reshape)
example2_3_1<-melt(example2_3,id.vars=c("id"),variable_name="运动员")
example2_3_1<-rename(example2_3_1,c(value="射击环数"))
save(example2_3_1,file="D:\\书籍资料整理\\统计R-贾俊平\\example2_3_1.RData")
head(example2_3_1);tail(example2_3_1)
dotchart(example2_3_1$射击环数,groups=example2_3_1$运动员,xlab="射击环数",pch=20)

#使用lattice包绘制的另一种形式的点图
library(lattice)
dotplot(射击环数~运动员,data=example2_3_1,col="blue",pch=19,font=0.5)

#核密度(密度分布)
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-1销售额.xlsx",sheetName = "Sheet1",encoding="UTF-8")
par(mfrow=c(1,2),cex=0.8,mai=c(0.7,0.7,0.1,0.1))
d<-table1_1$销售额
plot(density(d),col=1,xlab = "销售额",ylab="密度",main="")
rug(d,col="blue")
plot(density(d),xlab = "销售额",ylab="密度",main="")
polygon(density(d),col="gold",border="black")
rug(d,col="brown")

#lattic包绘制
dp1<-densityplot(~射击环数|运动员,example2_3_1,col="blue",cex=0.5,par.strip.text=list(cex=0.6),sub="(a)栅格图")
dp2<-densityplot(~射击环数,group=运动员,data=example2_3_1,auto.key=list(columns=1,x=0.01,y=0.95,cex=0.6),cex=0.5,sub="(b)比较图")
plot(dp1,split=c(1,1,2,1))
plot(dp2,split=c(2,1,2,1),newpage=F)

#用sm.density.compare函数绘制
attach(example2_3_1)
library(sm)
sm.density.compare(射击环数,运动员,lty=1:6,col=1:6)

#变量间关系
x<-seq(0,25,length=100)
y<-4+0.5*x+rnorm(100,0,2)
d<-data.frame(x,y)
plot(d)
polygon(d[chull(d),],col="pink",lty=3,lwd=2)
points(d)
abline(lm(y~x),lwd=2,col=4)
abline(y=mean(x),h=mean(y),lty=2,col="gray70")

#普通散点图和带有拟合直线的散点图
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-4医药企业.xlsx",sheetName = "Sheet1",encoding="UTF-8")
attach(table1_1)
par(mfcol=c(1,2),mai=c(0.7,0.7,0.3,0.1),cex=0.8,cex.main=0.8)
plot(广告费用,销售收入,main="(a)普通带网格线",type="n")
grid()
points(广告费用,销售收入,main="(a)普通带网格线")
rug(广告费用,side=1,col=4);rug(销售收入,side=2,col=4)
plot(广告费用,销售收入,main="(b)带有拟合直线")
abline(lm(销售收入~广告费用,data=table1_1),col="red")
rug(广告费用,side=1,col=4);rug(销售收入,side=2,col=4)

#带有两个变量线箱图的散点图
par(fig=c(0,0.8,0,0.8),mai=c(0.9,0.9,0.1,0.1))
plot(广告费用,销售收入,xlab="广告费用",ylab="销售收入",cex.lab=0.7,cex.axis=0.7)
abline(lm(销售收入~广告费用,data=table1_1),col="blue")
par(fig=c(0,0.8,0.5,1),new=TRUE)
boxplot(广告费用,horizontal = TRUE,axes=FALSE)
par(fig=c(0.52,1,0,0.9),new=TRUE)
boxplot(销售收入,axes=FALSE)

#重叠散点图
plot(广告费用,销售收入,xlab="",ylab="销售收入")
abline(lm(销售收入~广告费用,data=table1_1))
points(销售网点数,销售收入,pch=2,col="blue")
abline(lm(销售收入~销售网点数,data=table1_1),col="red")
points(销售人员数,销售收入,pch=3,col="blue")
abline(lm(销售收入~销售人员数,data=table1_1),col="red")
legend("bottomright",legend = c("广告费用","销售网点数","销售人员数"),pch=1:3,col=c("black","blue","red"))

#普通矩阵散点图
library(scatterplot3d)
plot(table1_1,cex=0.6,gap=0.5)
library(car)
library(carData)
attach(table1_1)
table1_1
scatterplotMatrix(~销售收入+销售网点数+销售人员数+广告费用,diagonal="histogram",gap=0.5)

#气泡图
n<-30;x<-rnorm(n);y<-rnorm(n);z<-abs(rnorm(n))+5:1
plot(x,y,cex=z,col="pink",pch=19)
points(x,y,cex=z)

#数据的气泡图
r<-sqrt(销售收入/pi)
symbols(广告费用,销售网点数,circles=r,inches=0.3,fg="white",bg="lightblue",ylab="销售网点数",xlab="广告费用")
text(广告费用,销售网点数,rownames(table1_1),cex=0.6)
mtext("气泡大小=销售收入",line=-2.5,adj=0.1)

#轮廓图
library(xlsx)
table1_1<-read.xlsx("D:\\书籍资料整理\\统计R-贾俊平\\2-5家庭收入.xlsx",sheetName = "Sheet1",encoding="UTF-8")
par(mai=c(0.7,0.7,0.1,0.1),cex=0.8)
matplot(t(table1_1[2:9]),type='b',lty=1:7,col=1:7,xlab = "消费项目",ylab="支出金额",pch=1,xaxt="n")
axis(side = 1,at=1:8,labels = c("食品","衣着","居住","家庭设备用品及服务","医疗保健","交通和通信","教育文化娱乐服务","其他商品和服务"),cex.axis=0.6)
legend(x="topright",legend = table1_1[,1],lty=1:7,col=1:7,text.width = 1,cex=0.7)