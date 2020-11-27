require(foreign)
require(MASS)
require(ResourceSelection)

table<-read.csv('D:/书籍资料整理/属性数据分析/政治意识与党派_整理数据.csv',encoding='UTF-8')
table$yishhi=factor(table$yishhi, ordered=T)
model<-polr(yishhi ~  dangpai ,data=table,method='logistic',Hess = TRUE)
summary(model)
ctable <- coef(summary(model))
(ctable <- coef(summary(model)))
#参数显著性检验
p <- pnorm(abs(ctable[,"t value"]),lower.tail = FALSE)*2
(ctable <- cbind(ctable,"p value"=p))

#参数的置信区间
(ci <- confint(model))
#优势
exp(coef(model))
exp(cbind(OR = coef(model), ci))
#没有实现拟合优度检验下边的方法不好使
#hl <- hoslem.test(table$yishhi,fitted(model),g=10)


