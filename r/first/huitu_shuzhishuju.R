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
#range=1.5,1.5倍的四分位差为内围栏,=3为外围栏(全包括)貌似与range=0极值一样的。varwidth=TRUE,箱子宽度与样本量有关。
par(mfrow=c(3,1),mai=c(0.4,0.2,0.3,0.2))
x<-rnorm(1000,50,5)
boxplot(x,range = 1.5,col="red",horizontal = TRUE,main="相邻值与箱子连线的线箱图range=1.5",cex=0.8)
boxplot(x,range = 3,col="green",horizontal = TRUE,main="相邻值与箱子连线的线箱图range=3",cex=0.8)
boxplot(x,range = 0,varwidth = TRUE,col="pink",horizontal = TRUE,main="极值与箱子连线的箱线图(range=0,varwidth=T)",cex=0.8)