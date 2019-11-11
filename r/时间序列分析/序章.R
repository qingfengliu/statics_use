#此例子为加州降雨量。来自潘书，历年线图。
library(TSA)
win.graph(width = 4.875,height =2.5,pointsize = 8)
data("larain")
plot(larain,ylab='Inches',xlab='Year',type = 'o')

#
win.graph(width = 3,height = 3,pointsize = 8)
plot(y=larain,x=zlag(larain),ylab = 'Inches',xlab='Previous Year Inches')